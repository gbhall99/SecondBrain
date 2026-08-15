/* SecondBrain shared page helpers — offline, no dependencies.
 * Loaded by base.html on every page as window.SB. See the UI-GUIDE comment in
 * secondbrain/web/templates/base.html for the full shell documentation.
 *
 *   SB.api(url, {method, body, button, headers, quiet})
 *       fetch wrapper: JSON-encodes a non-string body (and sets Content-Type),
 *       parses the JSON response, and on ANY failure — network error or a
 *       non-2xx status — shows an error toast (using the server's `detail`
 *       when present) and throws. Pass a button element as `button` to disable
 *       it and show a spinner while the request is in flight. Pass quiet:true
 *       to skip the toast when the caller renders the error inline instead.
 *       Thrown errors carry `.status` (HTTP status, or 0 for network errors).
 *       Every request carries an `X-SecondBrain: 1` header marking it as the
 *       app's own JS — the server refuses cross-origin writes without it (a
 *       drive-by page can't set custom headers), so keep using SB.api for
 *       anything that POSTs.
 *   SB.toast(message, kind)   transient message; kind: 'info'|'success'|'error'
 *   SB.reload(message, kind)  location.reload() that re-shows the toast after
 *                             the reload (for write flows that re-render) and
 *                             restores the scroll position, so a one-tap
 *                             action mid-page doesn't jump the user to the top.
 *   SB.busy(el, on)           manual busy-spinner toggle for a button.
 *   SB.esc(s)                 HTML-escape a string for innerHTML interpolation.
 *   SB.signout()              POST /logout (server revokes every outstanding
 *                             session cookie), then navigate to the login page
 *                             with a "signed out" confirmation.
 */
(function () {
  'use strict';

  var FLASH_KEY = 'sb-flash';

  function toastContainer() {
    var box = document.getElementById('toasts');
    if (!box) {
      box = document.createElement('div');
      box.id = 'toasts';
      box.setAttribute('aria-live', 'polite');
      document.body.appendChild(box);
    }
    return box;
  }

  function toast(message, kind) {
    kind = kind || 'info';
    var t = document.createElement('div');
    t.className = 'toast toast-' + kind;
    t.setAttribute('role', kind === 'error' ? 'alert' : 'status');
    t.textContent = String(message);
    t.title = 'Dismiss';
    toastContainer().appendChild(t);
    requestAnimationFrame(function () { t.classList.add('show'); });
    var timer = setTimeout(dismiss, kind === 'error' ? 6000 : 3200);
    t.addEventListener('click', dismiss);
    function dismiss() {
      clearTimeout(timer);
      t.classList.remove('show');
      setTimeout(function () { t.remove(); }, 250);
    }
    return t;
  }

  function busy(el, on) {
    if (!el || !el.classList) return;
    el.classList.toggle('busy', !!on);
    el.disabled = !!on;
    el.setAttribute('aria-busy', on ? 'true' : 'false');
  }

  async function api(url, opts) {
    opts = opts || {};
    // The custom header doubles as CSRF proof: cross-origin pages can't send
    // it (their preflight would be refused), so the server trusts our writes.
    var init = { method: opts.method || 'GET',
                 headers: Object.assign({ 'X-SecondBrain': '1' }, opts.headers) };
    if (opts.body !== undefined && opts.body !== null) {
      if (typeof opts.body === 'string') {
        init.body = opts.body;
      } else {
        init.body = JSON.stringify(opts.body);
        init.headers['Content-Type'] = 'application/json';
      }
    }
    busy(opts.button, true);
    try {
      var r;
      try {
        r = await fetch(url, init);
      } catch (netErr) {
        var offline = new Error('Network error — is SecondBrain running?');
        offline.status = 0;
        throw offline;
      }
      var text = await r.text();
      var data = null;
      if (text) {
        try { data = JSON.parse(text); } catch (parseErr) { data = null; }
      }
      if (!r.ok) {
        var msg = data && data.detail;
        if (msg && typeof msg !== 'string') msg = JSON.stringify(msg);
        var httpErr = new Error(msg || 'Request failed (HTTP ' + r.status + ')');
        httpErr.status = r.status;
        throw httpErr;
      }
      return data;
    } catch (err) {
      if (!opts.quiet) toast(err.message || 'Request failed', 'error');
      throw err;
    } finally {
      busy(opts.button, false);
    }
  }

  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function reload(message, kind) {
    try {
      sessionStorage.setItem(FLASH_KEY, JSON.stringify({
        message: message || '',
        kind: kind || 'success',
        // One-tap actions mid-way down a long page must not dump the user
        // back at the top: remember where they were and restore after reload.
        scrollY: Math.round(window.scrollY || 0),
        path: location.pathname + location.search
      }));
    } catch (e) { /* private mode: reload silently */ }
    location.reload();
  }

  async function signout() {
    try { await api('/logout', { method: 'POST' }); } catch (e) { /* already toasted */ }
    // If logout didn't stick (network error), /login sees the live session and
    // bounces straight home — so this never shows a false "signed out" notice.
    location.href = '/login?signedout=1';
  }

  // Re-show a toast queued by SB.reload() before the page reloaded, and put
  // the user back at the scroll position they acted from (same page only —
  // browsers don't reliably restore it when the reloaded content shifts).
  document.addEventListener('DOMContentLoaded', function () {
    var raw = null;
    try {
      raw = sessionStorage.getItem(FLASH_KEY);
      if (raw) sessionStorage.removeItem(FLASH_KEY);
    } catch (e) { /* storage unavailable */ }
    if (!raw) return;
    try {
      var f = JSON.parse(raw);
      if (!f) return;
      if (typeof f.scrollY === 'number' && f.scrollY > 0 &&
          f.path === location.pathname + location.search) {
        window.scrollTo(0, f.scrollY);
      }
      if (f.message) toast(f.message, f.kind || 'info');
    } catch (e) { /* ignore malformed flash */ }
  });

  /* ---- shared keyboard layer -------------------------------------------
   * '/'          focus the page's primary search (SB.registerSearch) or the
   *              nav search; 'g' then a letter navigates; '?' opens the help
   *              overlay rendered by base.html; Escape closes it.
   * All shortcuts are skipped while typing in a field.
   */
  var primarySearch = null;      // element or selector set by the page
  var GO = { h: '/', b: '/brief', t: '/tasks', d: '/decisions', y: '/day',
             l: '/timeline', p: '/relationships', r: '/projects', g: '/goals',
             a: '/chat' };
  var goArmed = 0;               // timestamp of a pending 'g' chord

  function registerSearch(elOrSel) { primarySearch = elOrSel; }

  function searchTarget() {
    var el = primarySearch;
    if (typeof el === 'string') el = document.querySelector(el);
    if (!el) el = document.getElementById('nav-q');
    return el;
  }

  function isTyping(t) {
    if (!t || !t.tagName) return false;
    var tag = t.tagName.toLowerCase();
    return tag === 'input' || tag === 'textarea' || tag === 'select' || t.isContentEditable;
  }

  function helpBox() { return document.getElementById('kbd-help'); }
  function openHelp() {
    var h = helpBox();
    if (!h) return;
    h.hidden = false;
    var btn = h.querySelector('.kbd-close');
    if (btn) btn.focus();
  }
  function closeHelp() {
    var h = helpBox();
    if (h) h.hidden = true;
  }

  document.addEventListener('keydown', function (e) {
    var h = helpBox();
    if (e.key === 'Escape' && h && !h.hidden) { closeHelp(); return; }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (isTyping(e.target)) return;
    if (e.key === '?') {
      e.preventDefault();
      if (h && h.hidden) openHelp(); else closeHelp();
      return;
    }
    if (e.key === '/') {
      var box = searchTarget();
      if (!box) return;
      e.preventDefault();
      // Reveal the collapsed nav search first on narrow screens.
      var wrap = box.closest ? box.closest('.nav-search') : null;
      if (wrap) wrap.classList.add('open');
      box.focus();
      if (box.select) box.select();
      return;
    }
    if (goArmed && Date.now() - goArmed < 1500) {
      goArmed = 0;
      var dest = GO[e.key];
      if (dest) { e.preventDefault(); location.href = dest; }
      return;
    }
    if (e.key === 'g' && !e.shiftKey) { goArmed = Date.now(); }
  });

  /* ---- list navigation (j/k + per-item action keys) --------------------
   * SB.listNav({ items: fn -> element[], keys: { d: fn(item), ... } })
   * j/k move a visible focus ring across the items; other registered keys
   * act on the focused item. Items get tabindex=-1 so they can hold focus.
   */
  function listNav(opts) {
    var focused = null;
    function items() {
      var els = typeof opts.items === 'function'
        ? opts.items()
        : document.querySelectorAll(opts.items);
      return Array.prototype.filter.call(els, function (el) {
        return !el.hidden && el.offsetParent !== null;
      });
    }
    function setFocus(el) {
      if (focused && focused !== el) focused.classList.remove('kbd-focus');
      focused = el;
      if (!el) return;
      el.classList.add('kbd-focus');
      el.setAttribute('tabindex', '-1');
      el.focus({ preventScroll: true });
      if (el.scrollIntoView) el.scrollIntoView({ block: 'nearest' });
    }
    document.addEventListener('keydown', function (e) {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (isTyping(e.target)) return;
      var list = items();
      if (!list.length) return;
      if (e.key === 'j' || e.key === 'k') {
        e.preventDefault();
        var idx = list.indexOf(focused);
        if (idx === -1) idx = e.key === 'j' ? -1 : 0;
        idx = e.key === 'j' ? Math.min(idx + 1, list.length - 1) : Math.max(idx - 1, 0);
        setFocus(list[idx]);
        return;
      }
      // An item counts as focused when the ring is on it OR focus is inside it.
      var cur = focused && list.indexOf(focused) !== -1 ? focused : null;
      if (!cur && e.target && e.target.closest) {
        cur = list.find ? list.find(function (el) { return el.contains(e.target); }) : null;
      }
      if (cur && opts.keys && opts.keys[e.key]) {
        e.preventDefault();
        opts.keys[e.key](cur);
      }
    });
    return { focus: setFocus, items: items };
  }

  /* ---- sortable tables (shared by Relationships / Projects) ------------
   * SB.sortableTable(table, { urlKey: 'sort' }) wires the .sortbtn column
   * headers: click to sort (text A→Z first; numbers/dates biggest first),
   * aria-sort kept truthful, and the current sort mirrored into the URL
   * (?<urlKey>=key.dir) so a re-sorted view survives reload/back.
   */
  function sortableTable(table, opts) {
    opts = opts || {};
    if (!table || !table.tBodies.length || !table.tHead) return;
    var tbody = table.tBodies[0];
    var rows = Array.prototype.slice.call(tbody.rows);
    var current = { key: null, dir: 0 };
    var urlKey = opts.urlKey || 'sort';

    function cellValue(tr, idx, type) {
      var td = tr.cells[idx];
      var v = td ? td.getAttribute('data-v') : null;
      if (v === null) v = td ? td.textContent : '';
      if (type === 'num') { var n = parseFloat(v); return isNaN(n) ? 0 : n; }
      return v.trim().toLowerCase();
    }

    function applySort(key, dir) {
      var btn = table.tHead.querySelector('.sortbtn[data-key="' + key + '"]');
      if (!btn) return;
      var th = btn.closest('th');
      var idx = th.cellIndex;
      var type = btn.getAttribute('data-type');
      current = { key: key, dir: dir };
      rows.sort(function (a, b) {
        var av = cellValue(a, idx, type), bv = cellValue(b, idx, type);
        if (av < bv) return -dir;
        if (av > bv) return dir;
        return 0;
      });
      rows.forEach(function (tr) { tbody.appendChild(tr); });
      Array.prototype.forEach.call(table.tHead.rows[0].cells, function (h) {
        h.removeAttribute('aria-sort');
      });
      th.setAttribute('aria-sort', dir === 1 ? 'ascending' : 'descending');
    }

    function syncURL() {
      try {
        var u = new URL(location.href);
        if (current.key) {
          u.searchParams.set(urlKey, current.key + '.' + (current.dir === 1 ? 'asc' : 'desc'));
        } else {
          u.searchParams.delete(urlKey);
        }
        history.replaceState(null, '', u);
      } catch (e) { /* keep the current URL */ }
    }

    table.tHead.addEventListener('click', function (e) {
      var btn = e.target.closest && e.target.closest('.sortbtn');
      if (!btn) return;
      var th = btn.closest('th');
      var type = btn.getAttribute('data-type');
      var key = btn.getAttribute('data-key');
      var dir = (current.key === key) ? -current.dir : (type === 'text' ? 1 : -1);
      applySort(key, dir);
      syncURL();
    });

    // Restore a bookmarked sort (?<urlKey>=key.asc|desc).
    try {
      var stored = new URLSearchParams(location.search).get(urlKey) || '';
      var m = /^([\w-]+)\.(asc|desc)$/.exec(stored);
      if (m) applySort(m[1], m[2] === 'asc' ? 1 : -1);
    } catch (e) { /* no restore */ }

    return { rows: rows, applySort: applySort };
  }

  /* ---- nav chrome ------------------------------------------------------ */
  document.addEventListener('DOMContentLoaded', function () {
    // The "More" overflow closes on outside click / Escape (a <details> would
    // otherwise stay open over the page).
    document.addEventListener('click', function (e) {
      document.querySelectorAll('details.nav-more[open]').forEach(function (d) {
        if (!d.contains(e.target)) d.open = false;
      });
    });
    document.addEventListener('keydown', function (e) {
      if (e.key !== 'Escape') return;
      document.querySelectorAll('details.nav-more[open]').forEach(function (d) {
        d.open = false;
      });
    });
    // Narrow screens: the nav search hides behind an icon toggle.
    var toggle = document.querySelector('.nav-search-toggle');
    if (toggle) {
      toggle.addEventListener('click', function () {
        var wrap = toggle.closest('.nav-search');
        var open = wrap.classList.toggle('open');
        toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
        if (open) {
          var box = document.getElementById('nav-q');
          if (box) box.focus();
        }
      });
    }
  });

  window.SB = { api: api, toast: toast, busy: busy, esc: esc, reload: reload,
                signout: signout, registerSearch: registerSearch,
                listNav: listNav, sortableTable: sortableTable,
                openHelp: openHelp, closeHelp: closeHelp };
})();
