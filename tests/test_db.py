import sqlite3

import pytest

from secondbrain.storage import db


def test_sqlite_module_is_stdlib_when_encryption_off(settings):
    assert settings.security.encrypt_db is False
    assert db._sqlite_module(settings) is sqlite3


def test_encrypt_db_without_driver_fails_clearly(settings):
    settings.security.encrypt_db = True
    settings.security.db_passphrase = "secret"
    if db.sqlcipher_available():
        pytest.skip("SQLCipher driver is installed; the failure path can't be exercised")
    with pytest.raises(RuntimeError, match="SQLCipher"):
        db.connect(settings=settings)


def test_sqlcipher_available_returns_bool():
    assert isinstance(db.sqlcipher_available(), bool)


def test_sql_string_literal_escapes_single_quotes():
    assert db._sql_string_literal("plain") == "plain"
    assert db._sql_string_literal("it's") == "it''s"
    assert db._sql_string_literal("'; DROP TABLE x; --") == "''; DROP TABLE x; --"


@pytest.mark.parametrize("passphrase", ["correct horse", "it's ' quoted", 'dq"and;semi--'])
def test_encrypted_db_round_trip_with_awkward_passphrases(settings, passphrase):
    """PRAGMAs can't take bound parameters, so the passphrase is inlined as a SQL
    literal; make sure quoting survives quotes/semicolons and the DB is unreadable
    without the key. Runs only when the [secure] driver is installed."""
    if not db.sqlcipher_available():
        pytest.skip("SQLCipher driver not installed (the [secure] extra)")
    settings.security.encrypt_db = True
    settings.security.db_passphrase = passphrase
    settings.ensure_dirs()
    c = db.init_db(settings=settings)
    c.execute("INSERT INTO speakers (id, name, kind, is_owner) VALUES (1, 'Me', 'owner', 1)")
    c.commit()
    c.close()

    plain = sqlite3.connect(str(settings.db_path))
    with pytest.raises(sqlite3.DatabaseError):
        plain.execute("SELECT name FROM sqlite_master").fetchall()
    plain.close()

    c2 = db.init_db(settings=settings)
    assert c2.execute("SELECT COUNT(*) FROM speakers").fetchone()[0] == 1
    c2.close()

    settings.security.db_passphrase = passphrase + "x"
    with pytest.raises(db.DB_ERRORS):
        db.init_db(settings=settings)
