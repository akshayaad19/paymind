// PayMind web app. Talks to the PayMind API with a JWT in the Authorization header.
// The token lives in sessionStorage (this tab only; gone when the tab closes).
// All permissions are enforced by the server; this page only decides what to show.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];
const TOKEN_KEY = "paymind.token";
const USER_KEY = "paymind.user";

const state = {
  token: sessionStorage.getItem(TOKEN_KEY),
  user: JSON.parse(sessionStorage.getItem(USER_KEY) || "null"),
  sessionId: null,
  busy: false,
  data: null,
  dataTab: null,
};

// ---------------------------------------------------------------- helpers

function esc(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Small, safe markdown: escape everything first, then allow **bold**, `code` and "- " / "* " lists.
function md(text) {
  const lines = esc(text).split("\n");
  let html = "", inList = false;
  for (const raw of lines) {
    const line = raw.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`([^`]+)`/g, "<code>$1</code>");
    const item = line.match(/^\s*[-*•]\s+(.*)/);
    if (item) { if (!inList) { html += "<ul>"; inList = true; } html += `<li>${item[1]}</li>`; continue; }
    if (inList) { html += "</ul>"; inList = false; }
    if (line.trim()) html += `<p>${line}</p>`;
  }
  return html + (inList ? "</ul>" : "");
}

function money(m) {
  if (!m) return "—";
  const n = Number(m.value);
  return `${n < 0 ? "−" : ""}$${Math.abs(n).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}
const shortDate = (iso) => (iso ? new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—");
const badge = (text, kind) => `<span class="badge ${esc(kind || text)}">${esc(String(text).replaceAll("_", " "))}</span>`;

function toast(text) {
  const t = $("#toast");
  t.textContent = text;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), 3500);
}

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  const res = await fetch(path, { ...options, headers });
  if (res.status === 401 && state.token) {
    logout("Your session has expired. Please log in again.");
    throw new Error("unauthorized");
  }
  const body = res.headers.get("content-type")?.includes("json") ? await res.json() : await res.text();
  if (!res.ok) throw new Error(body?.detail || `Request failed (${res.status})`);
  return body;
}

function tokenExpiry() {
  try { return new Date(JSON.parse(atob(state.token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/"))).exp * 1000); }
  catch { return null; }
}

// ---------------------------------------------------------------- auth

async function login(email, password) {
  $("#login-error").hidden = true;
  $("#login-btn").disabled = true;
  try {
    const res = await api("/api/auth/login", { method: "POST", body: JSON.stringify({ email, password }) });
    state.token = res.access_token;
    state.user = res.user;
    sessionStorage.setItem(TOKEN_KEY, state.token);
    sessionStorage.setItem(USER_KEY, JSON.stringify(state.user));
    showApp();
  } catch (err) {
    $("#login-error").textContent = err.message;
    $("#login-error").hidden = false;
  } finally {
    $("#login-btn").disabled = false;
  }
}

function logout(message) {
  state.token = null; state.user = null; state.sessionId = null; state.data = null;
  sessionStorage.removeItem(TOKEN_KEY);
  sessionStorage.removeItem(USER_KEY);
  $("#app-view").hidden = true;
  $("#login-view").hidden = false;
  if (message) { $("#login-error").textContent = message; $("#login-error").hidden = false; }
}

const SUGGESTIONS = {
  accountant: [
    "Is there a dispute open from user_123?",
    "What was my total sales volume last month?",
    "Send an invoice for $50 to john@x.com for 1 hour of consulting",
    "Which invoices are still unpaid?",
    "What can you do with invoices?",
    "What's the status of my last request?",
  ],
  customer: [
    "Show my open disputes",
    "Do I have any unpaid invoices?",
    "Send a message on my dispute saying I still haven't received the item",
    "Refund my last payment",
    "What can you help me with?",
  ],
};

function showApp() {
  const u = state.user;
  $("#login-view").hidden = true;
  $("#app-view").hidden = false;
  $("#user-avatar").textContent = u.name[0];
  $("#user-name").textContent = u.name;
  $("#user-role").textContent = u.role;
  $("#user-role").className = `badge ${u.role}`;
  $("#user-payer").textContent = u.payer_id ? `· ${u.payer_id}` : "";
  const exp = tokenExpiry();
  $("#token-note").textContent = exp ? `Signed in with a JWT · expires ${exp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}` : "";
  $("#suggestions").innerHTML = SUGGESTIONS[u.role].map((s) => `<li>${esc(s)}</li>`).join("");
  $("#data-reset").hidden = u.role !== "accountant";
  newChat();
  switchTab("chat");
  refreshStatus();
}

// ---------------------------------------------------------------- tabs

function switchTab(name) {
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === `panel-${name}`));
  if (name === "data") loadData();
  if (name === "audit") loadAudit();
  if (name === "chat") $("#input").focus();
}

// ---------------------------------------------------------------- chat

function newChat() {
  state.sessionId = null;
  $("#messages").innerHTML = "";
  $("#messages").append($("#chat-empty-template").content.cloneNode(true));
}

function addMessage(who, html, extraClass = "") {
  $("#chat-empty")?.remove();
  const el = document.createElement("div");
  el.className = `msg ${who}`;
  const avatar = who === "me" ? esc(state.user.name[0]) : "P";
  el.innerHTML = `<span class="avatar">${avatar}</span><div class="bubble ${extraClass}">${html}</div>`;
  $("#messages").append(el);
  el.scrollIntoView({ behavior: "smooth", block: "end" });
  return el;
}

function renderReply(res) {
  state.sessionId = res.session_id;
  if (res.confirmation) {
    const c = res.confirmation;
    const el = addMessage("bot", `
      <div class="confirm-card">
        <b>${c.large_amount ? "⚠️ Large amount: please check" : "Please confirm"}</b>
        <div class="q">${esc(c.question)}</div>
        <div class="buttons">
          <button class="btn btn-ok" data-approve="1">Yes, do it</button>
          <button class="btn btn-ghost" data-approve="0">No, cancel</button>
        </div>
      </div>`);
    el.querySelectorAll("[data-approve]").forEach((b) => b.addEventListener("click", () => answer(el, b.dataset.approve === "1")));
    return;
  }
  const failed = /can't reach the AI model/.test(res.reply || "");
  addMessage("bot", md(res.reply || "(no answer)"), failed ? "error" : "");
  if (failed) markStatus("llm", "warn");
}

async function withTyping(promise) {
  state.busy = true;
  $("#send").disabled = true;
  const typing = addMessage("bot", `<span class="typing"><span></span><span></span><span></span></span>`);
  try { return await promise; }
  finally { typing.remove(); state.busy = false; $("#send").disabled = false; }
}

async function send(text) {
  if (!text.trim() || state.busy) return;
  addMessage("me", esc(text));
  $("#input").value = "";
  autosize();
  try {
    const res = await withTyping(api("/api/chat", { method: "POST", body: JSON.stringify({ message: text, session_id: state.sessionId }) }));
    renderReply(res);
  } catch (err) {
    if (err.message !== "unauthorized") addMessage("bot", esc(err.message), "error");
  }
}

async function answer(cardEl, approve) {
  const card = cardEl.querySelector(".confirm-card");
  card.classList.add("done");
  card.querySelector(".buttons").innerHTML = approve ? badge("approved", "ok") : badge("cancelled", "declined");
  try {
    const res = await withTyping(api("/api/chat/confirm", { method: "POST", body: JSON.stringify({ session_id: state.sessionId, approve }) }));
    renderReply(res);
  } catch (err) {
    if (err.message !== "unauthorized") addMessage("bot", esc(err.message), "error");
  }
}

function autosize() {
  const t = $("#input");
  t.style.height = "auto";
  t.style.height = `${Math.min(t.scrollHeight, 160)}px`;
}

// ---------------------------------------------------------------- PayPal data

function buyerOf(d) { return d.disputed_transactions?.[0]?.buyer || {}; }
function recipientOf(inv) { const b = inv.primary_recipients?.[0]?.billing_info || {}; return b.email_address || "—"; }

const DATA_VIEWS = {
  disputes: {
    label: "Disputes",
    head: ["Dispute", "Buyer", "Reason", "Status", "Amount", "Opened"],
    row: (d) => [`<code>${esc(d.dispute_id)}</code>`, `${esc(buyerOf(d).name || "")}<div class="desc">${esc(buyerOf(d).payer_id || "")}</div>`,
      esc((d.reason || "").replaceAll("_", " ").toLowerCase()), badge(d.status, d.status === "RESOLVED" ? "ok" : "waiting"),
      `<span class="num">${money(d.dispute_amount)}</span>`, shortDate(d.create_time)],
  },
  invoices: {
    label: "Invoices",
    head: ["Invoice", "Recipient", "Status", "Amount", "Due", "Date"],
    row: (i) => [`<b>${esc(i.detail?.invoice_number)}</b><div class="desc mono">${esc(i.id)}</div>`, esc(recipientOf(i)),
      badge(i.status, ["PAID", "MARKED_AS_PAID"].includes(i.status) ? "ok" : i.status === "CANCELLED" ? "neutral" : "waiting"),
      money(i.amount), money(i.due_amount), esc(i.detail?.invoice_date)],
  },
  transactions: {
    label: "Transactions (30 days)",
    head: ["Transaction", "Type", "Customer", "Amount", "Fee", "Date"],
    row: (t) => { const i = t.transaction_info, p = t.payer_info || {}; const neg = Number(i.transaction_amount.value) < 0;
      return [`<code>${esc(i.transaction_id)}</code>`, badge(i.transaction_event_code === "T1107" ? "refund" : "sale", i.transaction_event_code === "T1107" ? "declined" : "ok"),
        esc(p.payer_name ? `${p.payer_name.given_name} ${p.payer_name.surname}` : "—"),
        `<span class="${neg ? "amount-neg" : "amount-pos"}">${money(i.transaction_amount)}</span>`, money(i.fee_amount), shortDate(i.transaction_initiation_date)]; },
  },
};

async function loadData() {
  $("#data-table").innerHTML = `<tr><td class="empty-row">Loading…</td></tr>`;
  try {
    state.data = await api("/api/paypal/overview");
  } catch (err) {
    $("#data-table").innerHTML = `<tr><td class="empty-row">${esc(err.message)}</td></tr>`;
    return;
  }
  const d = state.data;
  const open = d.disputes.filter((x) => x.status !== "RESOLVED").length;
  const unpaid = d.invoices.filter((x) => ["SENT", "PARTIALLY_PAID", "UNPAID"].includes(x.status));
  if (d.role === "accountant") {
    const sales = d.transactions.filter((t) => t.transaction_info.transaction_event_code === "T0006")
      .reduce((s, t) => s + Number(t.transaction_info.transaction_amount.value), 0);
    $("#data-sub").textContent = "The whole shop account, read through the same PayPal APIs the agent uses.";
    $("#stats").innerHTML = [
      ["Balance", money(d.balance)], ["Sales, last 30 days", money({ value: sales })],
      ["Open disputes", open], ["Unpaid invoices", unpaid.length],
    ].map(([l, v]) => `<div class="stat"><div class="label">${l}</div><div class="value">${v}</div></div>`).join("");
  } else {
    $("#data-sub").textContent = `Only your own records (${state.user.payer_id}). Other customers' data is filtered out by the server.`;
    $("#stats").innerHTML = [["My open disputes", open], ["My unpaid invoices", unpaid.length]]
      .map(([l, v]) => `<div class="stat"><div class="label">${l}</div><div class="value">${v}</div></div>`).join("");
  }
  const views = Object.keys(DATA_VIEWS).filter((k) => Array.isArray(d[k]));
  if (!views.includes(state.dataTab)) state.dataTab = views[0];
  $("#data-subtabs").innerHTML = views.map((k) => `<button data-view="${k}" class="${k === state.dataTab ? "on" : ""}">${DATA_VIEWS[k].label} · ${d[k].length}</button>`).join("");
  renderDataTable();
}

function renderDataTable() {
  const view = DATA_VIEWS[state.dataTab], rows = state.data[state.dataTab];
  $("#data-table").innerHTML = `<tr>${view.head.map((h) => `<th>${h}</th>`).join("")}</tr>` +
    (rows.map((r) => `<tr>${view.row(r).map((c) => `<td>${c}</td>`).join("")}</tr>`).join("") ||
      `<tr><td class="empty-row" colspan="${view.head.length}">Nothing here.</td></tr>`);
}

// ---------------------------------------------------------------- audit

async function loadAudit() {
  $("#audit-table").innerHTML = `<tr><td class="empty-row">Loading…</td></tr>`;
  try {
    const res = await api("/api/audit");
    $("#audit-table").innerHTML = `<tr><th>Time</th><th>Tool</th><th>Details</th><th>Status</th><th>HTTP</th><th>Confirmed</th></tr>` +
      (res.items.map((a) => `<tr>
        <td>${shortDate(a.time)}</td><td><code>${esc(a.tool)}</code></td>
        <td><div class="desc mono">${esc(JSON.stringify(a.params)).slice(0, 160)}</div>${a.result_summary ? `<div class="desc">${esc(a.result_summary)}</div>` : ""}</td>
        <td>${badge(a.status)}</td><td>${a.http_status ?? "—"}</td><td>${a.confirmed ? "✅" : ""}</td></tr>`).join("") ||
        `<tr><td class="empty-row" colspan="6">No actions yet. Ask the assistant something in Chat.</td></tr>`);
  } catch (err) {
    $("#audit-table").innerHTML = `<tr><td class="empty-row">${esc(err.message)}</td></tr>`;
  }
}

// ---------------------------------------------------------------- status

function markStatus(key, kind) {
  const li = $(`#status li[data-key="${key}"]`);
  if (li) li.className = kind;
}

async function refreshStatus() {
  try {
    const s = await api("/api/health");
    for (const [key, v] of Object.entries(s)) {
      markStatus(key, v.ok ? "ok" : key === "llm" ? "warn" : "bad");
      const li = $(`#status li[data-key="${key}"]`);
      if (li) li.title = v.detail;
    }
  } catch { /* shown by other errors */ }
}

// ---------------------------------------------------------------- wiring

document.addEventListener("DOMContentLoaded", () => {
  // keep a copy of the empty chat state for "New chat"
  const tpl = document.createElement("template");
  tpl.id = "chat-empty-template";
  tpl.content.append($("#chat-empty").cloneNode(true));
  document.body.append(tpl);

  $("#login-form").addEventListener("submit", (e) => { e.preventDefault(); login($("#login-email").value, $("#login-password").value); });
  $$(".demo").forEach((b) => b.addEventListener("click", () => {
    $("#login-email").value = b.dataset.email;
    $("#login-password").value = b.dataset.password;
    login(b.dataset.email, b.dataset.password);
  }));
  $("#logout").addEventListener("click", () => logout());
  $("#new-chat").addEventListener("click", () => { newChat(); switchTab("chat"); });
  $$(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));
  $("#suggestions").addEventListener("click", (e) => { if (e.target.tagName === "LI") { switchTab("chat"); send(e.target.textContent); } });

  $("#composer").addEventListener("submit", (e) => { e.preventDefault(); send($("#input").value); });
  $("#input").addEventListener("input", autosize);
  $("#input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send($("#input").value); } });

  $("#data-refresh").addEventListener("click", loadData);
  $("#data-subtabs").addEventListener("click", (e) => {
    if (!e.target.dataset.view || !state.data) return;
    state.dataTab = e.target.dataset.view;
    $$("#data-subtabs button").forEach((b) => b.classList.toggle("on", b.dataset.view === state.dataTab));
    renderDataTable();
  });
  $("#data-reset").addEventListener("click", async () => {
    if (!confirmReset()) return;
    try { await api("/api/paypal/reset", { method: "POST" }); toast("Demo data restored to the start."); loadData(); }
    catch (err) { if (err.message !== "unauthorized") toast(err.message); }
  });
  $("#audit-refresh").addEventListener("click", loadAudit);

  if (state.token && state.user) {
    api("/api/me").then((u) => { state.user = u; showApp(); }).catch(() => logout());
  } else {
    $("#login-view").hidden = false;
  }
});

// A second click within 4 seconds confirms the reset (no browser pop-ups).
function confirmReset() {
  const btn = $("#data-reset");
  if (btn.dataset.armed) { delete btn.dataset.armed; btn.textContent = "Reset demo data"; return true; }
  btn.dataset.armed = "1";
  btn.textContent = "Click again to reset";
  setTimeout(() => { delete btn.dataset.armed; btn.textContent = "Reset demo data"; }, 4000);
  return false;
}
