// PayMind web app. Talks to the PayMind API with a JWT in the Authorization header.
// The token is kept only in memory: refreshing or closing the page logs you out,
// so users log in every time. All permissions are enforced by the server; this
// page only decides what to show.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

const state = {
  token: null,
  user: null,
  sessionId: null,
  busy: false,
  data: null,
  dataTab: null,
  txPeriod: "month",
  txAnchor: new Date(),
  tx: null,
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

// ---------------------------------------------------------------- auth

async function login(email, password) {
  $("#login-error").hidden = true;
  $("#login-btn").disabled = true;
  try {
    const res = await api("/api/auth/login", { method: "POST", body: JSON.stringify({ email, password }) });
    state.token = res.access_token;
    state.user = res.user;
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
  $("#drawer").hidden = $("#drawer-backdrop").hidden = true;
  $("#app-view").hidden = true;
  $("#login-view").hidden = false;
  if (message) { $("#login-error").textContent = message; $("#login-error").hidden = false; }
}

const SUGGESTIONS = {
  accountant: [
    "Show open disputes",
    "Which invoices are unpaid?",
    "How were sales this month?",
    "Create an invoice",
    "Any new messages?",
  ],
  customer: [
    "Show my disputes",
    "Do I have unpaid invoices?",
    "Any reply from the shop?",
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
  $("#suggestions").innerHTML = SUGGESTIONS[u.role].map((s) => `<li>${esc(s)}</li>`).join("");
  newChat();
  switchTab("chat");
  loadWhatsNew();
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

// Dispute status and reason in plain words, from the viewer's side (shop or customer).
const DISPUTE_REASONS = {
  MERCHANDISE_OR_SERVICE_NOT_RECEIVED: "Item not received",
  MERCHANDISE_OR_SERVICE_NOT_AS_DESCRIBED: "Item not as described",
  UNAUTHORISED: "Payment not authorised",
  CREDIT_NOT_PROCESSED: "Refund not received",
  DUPLICATE_TRANSACTION: "Charged twice",
  INCORRECT_AMOUNT: "Wrong amount charged",
};
function disputeReason(d) { return DISPUTE_REASONS[d.reason] || String(d.reason || "").replaceAll("_", " ").toLowerCase(); }
function disputeStatus(d) {
  const shop = state.user.role === "accountant";
  switch (d.status) {
    case "WAITING_FOR_SELLER_RESPONSE": return shop ? { text: "Needs your response", kind: "error" } : { text: "Waiting for the shop", kind: "waiting" };
    case "WAITING_FOR_BUYER_RESPONSE": return shop ? { text: "Waiting for the customer", kind: "waiting" } : { text: "Needs your response", kind: "error" };
    case "UNDER_REVIEW": return { text: "PayPal is reviewing", kind: "waiting" };
    case "RESOLVED": return { text: "Resolved", kind: "ok" };
    default: return { text: String(d.status || "").replaceAll("_", " ").toLowerCase(), kind: "neutral" };
  }
}

// PayPal's invoice statuses in plain words. SENT means "sent and waiting for payment".
function invoiceStatus(inv) {
  const due = inv.detail?.payment_term?.due_date;
  const today = new Date().toISOString().slice(0, 10);
  switch (inv.status) {
    case "DRAFT": return { text: "Draft", kind: "neutral" };
    case "SENT": case "UNPAID": case "SCHEDULED":
      return due && due < today ? { text: "Overdue", kind: "error" } : { text: "Awaiting payment", kind: "waiting" };
    case "PARTIALLY_PAID": return { text: "Partly paid", kind: "waiting" };
    case "PAID": case "MARKED_AS_PAID": return { text: "Paid", kind: "ok" };
    case "CANCELLED": return { text: "Cancelled", kind: "neutral" };
    default: return { text: String(inv.status || "").replaceAll("_", " ").toLowerCase(), kind: "neutral" };
  }
}

function buyerOf(d) { return d.disputed_transactions?.[0]?.buyer || {}; }
function recipientOf(inv) { const b = inv.primary_recipients?.[0]?.billing_info || {}; return b.email_address || "—"; }

const DATA_VIEWS = {
  disputes: {
    label: "Disputes",
    head: ["Dispute", "Buyer", "Reason", "Status", "Amount", "Opened"],
    rowAttrs: (d) => `class="clickable" data-dispute="${esc(d.dispute_id)}"`,
    row: (d) => { const s = disputeStatus(d);
      return [`<code>${esc(d.dispute_id)}</code>${d.unread ? `<span class="new-badge">${d.unread} new</span>` : ""}`,
      `${esc(buyerOf(d).name || "")}${state.user.role === "accountant" ? `<div class="desc">${esc(buyerOf(d).payer_id || "")}</div>` : ""}`,
      esc(disputeReason(d)), `<span class="badge ${s.kind}">${esc(s.text)}</span>`,
      `<span class="num">${money(d.dispute_amount)}</span>`, shortDate(d.create_time)]; },
  },
  invoices: {
    label: "Invoices",
    head: ["Invoice", "Recipient", "Status", "Amount", "Amount due", "Due date", "Issued"],
    row: (i) => { const s = invoiceStatus(i);
      return [`<b>${esc(i.detail?.invoice_number)}</b><div class="desc mono">${esc(i.id)}</div>`, esc(recipientOf(i)),
        `<span class="badge ${s.kind}">${esc(s.text)}</span>`, money(i.amount), money(i.due_amount),
        esc(i.detail?.payment_term?.due_date || "—"), esc(i.detail?.invoice_date)]; },
  },
  transactions: {
    label: "Transactions",
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
    $("#data-sub").textContent = "Your shop's account at a glance: balance, sales, disputes and invoices.";
    $("#stats").innerHTML = [
      ["Balance", money(d.balance)], ["Sales, last 30 days", money({ value: sales })],
      ["Open disputes", open], ["Unpaid invoices", unpaid.length],
    ].map(([l, v]) => `<div class="stat"><div class="label">${l}</div><div class="value">${v}</div></div>`).join("");
  } else {
    $("#data-sub").textContent = "Your disputes and invoices with this shop.";
    $("#stats").innerHTML = [["My open disputes", open], ["My unpaid invoices", unpaid.length]]
      .map(([l, v]) => `<div class="stat"><div class="label">${l}</div><div class="value">${v}</div></div>`).join("");
  }
  const views = Object.keys(DATA_VIEWS).filter((k) => Array.isArray(d[k]));
  if (!views.includes(state.dataTab)) state.dataTab = views[0];
  $("#data-subtabs").innerHTML = views.map((k) => `<button data-view="${k}" class="${k === state.dataTab ? "on" : ""}">${DATA_VIEWS[k].label}${k === "transactions" ? "" : ` · ${d[k].length}`}</button>`).join("");
  renderDataTable();
}

// ---------- transactions: day / week / month in the user's local time ----------

function periodRange(period, anchor) {
  const a = new Date(anchor.getFullYear(), anchor.getMonth(), anchor.getDate());
  let start, end;
  if (period === "day") { start = a; end = new Date(a); end.setDate(a.getDate() + 1); }
  else if (period === "week") { start = new Date(a); start.setDate(a.getDate() - ((a.getDay() + 6) % 7)); end = new Date(start); end.setDate(start.getDate() + 7); }
  else { start = new Date(a.getFullYear(), a.getMonth(), 1); end = new Date(a.getFullYear(), a.getMonth() + 1, 1); }
  return { start, end: new Date(end.getTime() - 1000) };  // end = last second of the period
}

function periodLabel(period, { start, end }) {
  const d = (x, o) => x.toLocaleDateString(undefined, o);
  if (period === "day") return d(start, { weekday: "short", day: "numeric", month: "short", year: "numeric" });
  if (period === "week") return `${d(start, { day: "numeric", month: "short" })} – ${d(end, { day: "numeric", month: "short", year: "numeric" })}`;
  return d(start, { month: "long", year: "numeric" });
}

function localIso(date) {  // 2026-09-01T00:00:00+05:30
  const pad = (n) => String(Math.abs(n)).padStart(2, "0");
  const off = -date.getTimezoneOffset();
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}` +
    `${off >= 0 ? "+" : "-"}${pad(Math.floor(Math.abs(off) / 60))}:${pad(Math.abs(off) % 60)}`;
}

function shiftPeriod(step) {
  const a = new Date(state.txAnchor);
  if (state.txPeriod === "day") a.setDate(a.getDate() + step);
  else if (state.txPeriod === "week") a.setDate(a.getDate() + 7 * step);
  else a.setMonth(a.getMonth() + step, 1);
  state.txAnchor = a;
  loadTransactions();
}

async function loadTransactions() {
  const range = periodRange(state.txPeriod, state.txAnchor);
  $("#tx-label").textContent = periodLabel(state.txPeriod, range);
  $$("#tx-period button").forEach((b) => b.classList.toggle("on", b.dataset.period === state.txPeriod));
  $("#tx-next").disabled = range.end >= new Date();  // nothing in the future
  $("#data-table").innerHTML = `<tr><td class="empty-row">Loading…</td></tr>`;
  try {
    state.tx = await api(`/api/paypal/transactions?start=${encodeURIComponent(localIso(range.start))}&end=${encodeURIComponent(localIso(range.end))}`);
  } catch (err) {
    if (err.message !== "unauthorized") $("#data-table").innerHTML = `<tr><td class="empty-row">${esc(err.message)}</td></tr>`;
    return;
  }
  const t = state.tx.totals;
  $("#tx-totals").innerHTML = [
    ["Sales", `<span class="amount-pos">${money({ value: t.sales })}</span>`],
    ["Refunds", `<span class="amount-neg">${money({ value: t.refunds })}</span>`],
    ["Fees", money({ value: t.fees })],
    ["Net", money({ value: t.net })],
    ["Transactions", t.count],
  ].map(([l, v]) => `<span class="chip">${l}<b>${v}</b></span>`).join("");
  renderRows(DATA_VIEWS.transactions, state.tx.transactions, "No transactions in this period.");
}

function renderRows(view, rows, emptyText = "Nothing here.") {
  $("#data-table").innerHTML = `<tr>${view.head.map((h) => `<th>${h}</th>`).join("")}</tr>` +
    (rows.map((r) => `<tr ${view.rowAttrs ? view.rowAttrs(r) : ""}>${view.row(r).map((c) => `<td>${c}</td>`).join("")}</tr>`).join("") ||
      `<tr><td class="empty-row" colspan="${view.head.length}">${emptyText}</td></tr>`);
}

function renderDataTable() {
  const isTx = state.dataTab === "transactions";
  $("#tx-toolbar").hidden = !isTx;
  $("#tx-totals").hidden = !isTx;
  if (isTx) return loadTransactions();
  const view = DATA_VIEWS[state.dataTab], rows = state.data[state.dataTab];
  renderRows(view, rows);
}

// ---------------------------------------------------------------- what's new (top of chat, no AI needed)

function timeAgo(iso) {
  const mins = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
  if (mins < 60) return mins <= 1 ? "just now" : `${mins} minutes ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours} hour${hours > 1 ? "s" : ""} ago`;
  const days = Math.round(hours / 24);
  return days === 1 ? "yesterday" : `${days} days ago`;
}

function whatsNewItem(i) {
  const about = `${esc(DISPUTE_REASONS[i.reason] || "Dispute")} · ${money(i.amount)}`;
  const quote = i.text ? `<div class="quote">“${esc(i.text.length > 140 ? i.text.slice(0, 140) + "…" : i.text)}”</div>` : "";
  const due = !i.action_needed ? "" : i.days_left === null ? " · action needed"
    : i.days_left < 0 ? ` · <span class="amount-neg">overdue by ${-i.days_left} day${i.days_left === -1 ? "" : "s"}</span>`
    : ` · respond by ${esc(new Date(i.due_date).toLocaleDateString(undefined, { day: "numeric", month: "short" }))} (${i.days_left === 0 ? "today" : `${i.days_left} day${i.days_left === 1 ? "" : "s"} left`})`;
  const texts = {
    action_needed: [`🔴 Action needed · ${esc(i.with)}${due}`, "Open", "open"],
    new_message: [`📬 New message from ${esc(i.with)} · ${timeAgo(i.time)}${due}`, "Open", "open"],
    needs_reply: [`✍️ ${esc(i.with)} wrote ${timeAgo(i.time)} · waiting for your reply${due}`, "Reply", "open"],
    no_reply_yet: [`⏳ You wrote ${timeAgo(i.time)} · no reply yet from ${esc(i.with)}`, "Send a reminder", "remind"],
  };
  const [what, label, action] = texts[i.kind];
  return `<div class="wn-item"><div><div class="what">${what}</div><div class="quote">${about}</div>${quote}</div>
    <button class="btn ${action === "remind" ? "btn-primary" : "btn-ghost"}" data-wn="${action}" data-dispute="${esc(i.dispute_id)}"
      data-days="${i.days || ""}">${label}</button></div>`;
}

async function loadWhatsNew() {
  let res;
  try { res = await api("/api/whats-new"); } catch { return; }
  if (!res.items.length || state.sessionId) return;  // nothing to show, or the chat already started
  $("#chat-empty")?.remove();
  const card = document.createElement("div");
  card.className = "whats-new";
  card.id = "whats-new";
  card.innerHTML = `<h3>What's new</h3>${res.items.map(whatsNewItem).join("")}`;
  $("#messages").prepend(card);
  card.addEventListener("click", (e) => {
    const b = e.target.closest("[data-wn]");
    if (!b) return;
    if (b.dataset.wn === "open") return openDispute(b.dataset.dispute);
    const days = b.dataset.days ? ` I wrote ${b.dataset.days} days ago and haven't heard back.` : "";
    send(`Send a polite reminder on dispute ${b.dataset.dispute}.${days}`);
  });
}

// ---------------------------------------------------------------- dispute conversation

let openDisputeId = null;

function renderThread(res) {
  const d = res.dispute, s = disputeStatus(d), mine = state.user.role === "accountant" ? "SELLER" : "BUYER";
  $("#drawer-title").textContent = `${disputeReason(d)} · ${money(d.dispute_amount)}`;
  $("#drawer-sub").innerHTML = `<span class="mono">${esc(d.dispute_id)}</span><span class="badge ${s.kind}">${esc(s.text)}</span>` +
    (state.user.role === "accountant" ? `<span>with ${esc(buyerOf(d).name || "customer")}</span>` : "");
  $("#thread").innerHTML = res.messages.map((m) => `
    <div class="tmsg ${m.from === mine ? "mine" : ""}">
      <div class="who">${esc(m.from === mine ? "You" : m.name)} · ${shortDate(m.time)}</div>
      <div class="text">${esc(m.text)}</div>
    </div>`).join("") || `<p class="muted">No messages yet.</p>`;
  $("#thread").scrollTop = $("#thread").scrollHeight;
  $("#thread-form").hidden = !res.can_reply;
  $("#thread-closed").hidden = res.can_reply;
}

async function openDispute(id) {
  openDisputeId = id;
  $("#drawer").hidden = $("#drawer-backdrop").hidden = false;
  $("#thread").innerHTML = `<p class="muted">Loading…</p>`;
  try { renderThread(await api(`/api/disputes/${encodeURIComponent(id)}`)); $("#thread-input").focus(); }
  catch (err) { if (err.message !== "unauthorized") $("#thread").innerHTML = `<p class="muted">${esc(err.message)}</p>`; }
}

function closeDispute() {
  $("#drawer").hidden = $("#drawer-backdrop").hidden = true;
  openDisputeId = null;
  loadData();  // refresh the "new" badges
}

async function sendThreadMessage() {
  const text = $("#thread-input").value.trim();
  if (!text || !openDisputeId) return;
  $("#thread-send").disabled = true;
  try {
    renderThread(await api(`/api/disputes/${encodeURIComponent(openDisputeId)}/messages`, { method: "POST", body: JSON.stringify({ message: text }) }));
    $("#thread-input").value = "";
  } catch (err) { if (err.message !== "unauthorized") toast(err.message); }
  finally { $("#thread-send").disabled = false; }
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

// ---------------------------------------------------------------- wiring

document.addEventListener("DOMContentLoaded", () => {
  // keep a copy of the empty chat state for "New chat"
  const tpl = document.createElement("template");
  tpl.id = "chat-empty-template";
  tpl.content.append($("#chat-empty").cloneNode(true));
  document.body.append(tpl);

  $("#login-form").addEventListener("submit", (e) => { e.preventDefault(); login($("#login-email").value, $("#login-password").value); });
  $("#logout").addEventListener("click", () => logout());
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
  $("#audit-refresh").addEventListener("click", loadAudit);
  $("#data-table").addEventListener("click", (e) => { const row = e.target.closest("tr[data-dispute]"); if (row) openDispute(row.dataset.dispute); });
  $("#drawer-close").addEventListener("click", closeDispute);
  $("#drawer-backdrop").addEventListener("click", closeDispute);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("#drawer").hidden) closeDispute(); });
  $("#thread-form").addEventListener("submit", (e) => { e.preventDefault(); sendThreadMessage(); });
  $("#thread-input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendThreadMessage(); } });
  $$("#tx-period button").forEach((b) => b.addEventListener("click", () => { state.txPeriod = b.dataset.period; state.txAnchor = new Date(); loadTransactions(); }));
  $("#tx-prev").addEventListener("click", () => shiftPeriod(-1));
  $("#tx-next").addEventListener("click", () => shiftPeriod(1));

  $("#login-view").hidden = false;  // always start at login
});
