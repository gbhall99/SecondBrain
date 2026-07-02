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
