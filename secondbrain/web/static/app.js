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

  window.SB = { api: api, toast: toast, busy: busy, esc: esc, reload: reload, signout: signout };
})();
