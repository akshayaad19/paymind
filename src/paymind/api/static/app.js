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
  // Clicking "Log out" also tells the server, so this token (and any copy of it) stops working.
  // After a 401 (message given) the token is already dead: nothing to tell.
  if (state.token && !message) {
    fetch("/api/auth/logout", { method: "POST", headers: { Authorization: `Bearer ${state.token}` } }).catch(() => {});
  }
  state.token = null; state.user = null; state.sessionId = null; state.data = null; state.customerFilter = "";
  $("#customer-filter").value = "";
  $("#drawer").hidden = $("#drawer-backdrop").hidden = true;
  $("#po-drawer").hidden = $("#po-backdrop").hidden = true;
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
  $("#orders-actions").hidden = u.role !== "customer";
  $("#orders-sub").textContent = u.role === "customer"
    ? "Send the shop a purchase order: upload a photo or scan (handwritten is fine) or type it in. You check everything before it's sent."
    : "Purchase orders from customers. Review each one, set prices and a delivery date to accept (this creates an invoice), or reject it.";
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
  if (name === "orders") loadOrders();
  if (name === "chat") { $("#input").focus(); loadWhatsNew(); }
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

function renderReply(res, streamed = null) {
  state.sessionId = res.session_id;
  if (res.confirmation) {
    streamed?.remove();  // any words written before the pause are replaced by the confirm card
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
  // The receipt comes from the server's record of what actually ran, not from the AI's wording.
  const r = res.receipt;
  const receiptHtml = r ? `<div class="receipt ${r.changes.some((c) => c.outcome === "done") ? "changed" : ""}">🧾 ${esc(r.summary)}</div>` : "";
  const html = md(res.reply || "(no answer)") + receiptHtml;
  if (streamed) {  // the streamed bubble becomes the final answer (same text, now with the receipt)
    const bubble = streamed.querySelector(".bubble");
    bubble.innerHTML = html;
    bubble.classList.toggle("error", failed);
  } else {
    addMessage("bot", html, failed ? "error" : "");
  }
  loadWhatsNew();
}

// Streams a chat request: the "…" dots show until the first words arrive, then the answer is
// typed into its bubble as Gemini writes it. Gemini sends a sentence or so at a time, very fast,
// so received text goes into a queue that is typed out a few characters per frame (faster when
// a lot is waiting). The final "done" event carries the full result (reply, confirmation,
// receipt), which replaces the streamed text once the typing has caught up.
async function streamChat(path, body) {
  state.busy = true;
  $("#send").disabled = true;
  const typing = addMessage("bot", `<span class="typing"><span></span><span></span><span></span></span>`);
  let bubble = null, text = "", shown = 0, final = null, ticking = null;
  const tick = () => {
    if (shown >= text.length) { ticking = null; return; }
    if (!bubble) { typing.remove(); bubble = addMessage("bot", ""); }
    shown = Math.min(text.length, shown + Math.max(2, Math.ceil((text.length - shown) / 30)));
    bubble.querySelector(".bubble").innerHTML = md(text.slice(0, shown));
    bubble.scrollIntoView({ block: "end" });
    ticking = requestAnimationFrame(tick);
  };
  const caughtUp = () => new Promise((resolve) => {
    const wait = () => (shown >= text.length ? resolve() : setTimeout(wait, 30));
    wait();
  });
  try {
    const res = await fetch(path, { method: "POST", body: JSON.stringify(body),
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${state.token}` } });
    if (res.status === 401) { logout("Your session has expired. Please log in again."); throw new Error("unauthorized"); }
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `Request failed (${res.status})`);
    const reader = res.body.getReader(), decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop();
      for (const raw of events) {
        if (!raw.startsWith("data: ")) continue;
        const ev = JSON.parse(raw.slice(6));
        if (ev.type === "error") throw new Error(ev.detail);
        if (ev.type === "done") { final = ev; continue; }
        if (ev.type === "restart") { text = ""; shown = 0; continue; }
        if (ev.type === "token") {
          text += ev.text;
          if (!ticking) ticking = requestAnimationFrame(tick);
        }
      }
    }
    if (!final) throw new Error("The answer was cut off. Please try again.");
    if (final.reply && !final.confirmation) await caughtUp();
    return { final, bubble };
  } finally {
    if (ticking) cancelAnimationFrame(ticking);
    typing.remove();
    state.busy = false;
    $("#send").disabled = false;
  }
}

async function send(text) {
  if (!text.trim() || state.busy) return;
  addMessage("me", esc(text));
  $("#input").value = "";
  autosize();
  try {
    const out = await streamChat("/api/chat/stream", { message: text, session_id: state.sessionId });
    renderReply(out.final, out.bubble);
  } catch (err) {
    if (err.message !== "unauthorized") addMessage("bot", esc(err.message), "error");
  }
}

async function answer(cardEl, approve) {
  const card = cardEl.querySelector(".confirm-card");
  card.classList.add("done");
  card.querySelector(".buttons").innerHTML = approve ? badge("approved", "ok") : badge("cancelled", "declined");
  try {
    const out = await streamChat("/api/chat/confirm/stream", { session_id: state.sessionId, approve });
    renderReply(out.final, out.bubble);
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
  const ev = d.evidence?.status;
  if (ev && d.status !== "RESOLVED") {
    const label = shop
      ? { requested: ["Waiting for photos", "waiting"], rejected: ["Waiting for photos", "waiting"], submitted: ["Check photos", "error"], approved: ["Send replacement", "error"] }[ev]
      : { requested: ["Photo needed", "error"], rejected: ["Photo needed", "error"], submitted: ["Photos under review", "waiting"], approved: ["Replacement being arranged", "waiting"] }[ev];
    if (label) return { text: label[0], kind: label[1] };
  }
  if (d.refund_request) {
    const what = d.refund_request.wants === "replacement" ? "Replacement" : "Refund";
    return shop ? { text: `${what} requested by customer`, kind: "error" } : { text: `${what} requested`, kind: "waiting" };
  }
  if (d.seller_action && d.status === "WAITING_FOR_BUYER_RESPONSE") {
    const what = d.seller_action.type === "replacement" ? "Replacement sent" : "Refunded";
    return shop ? { text: `${what} · waiting for customer`, kind: "waiting" } : { text: `${what} · please confirm`, kind: "error" };
  }
  switch (d.status) {
    case "WAITING_FOR_SELLER_RESPONSE": return shop ? { text: "Needs your response", kind: "error" } : { text: "Waiting for the shop", kind: "waiting" };
    case "WAITING_FOR_BUYER_RESPONSE": return shop ? { text: "Waiting for the customer", kind: "waiting" } : { text: "Needs your response", kind: "error" };
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
function recipientName(inv) { const n = inv.primary_recipients?.[0]?.billing_info?.name || {}; return [n.given_name, n.surname].filter(Boolean).join(" "); }
function payerName(t) { const n = t.payer_info?.payer_name; return n ? `${n.given_name} ${n.surname}` : ""; }
function invoiceNumber(id) { return state.data?.invoices?.find((i) => i.id === id)?.detail?.invoice_number || id; }
// How a payment was made, from PayPal's own fields: invoice_id → paid an invoice,
// store_info → paid at the shop's till, otherwise → online store checkout.
function paidHow(t) {
  const i = t.transaction_info;
  if (i.transaction_event_code === "T1107") return `<span class="muted">Refund to customer</span>`;
  if (i.invoice_id) return `🧾 Invoice <b>${esc(invoiceNumber(i.invoice_id))}</b>`;
  if (t.store_info) return `🏬 In store<div class="desc">${esc(t.store_info.store_id)} · till ${esc(t.store_info.terminal_id)}</div>`;
  return `🌐 Online store`;
}
function paidOn(inv) { return inv.payments?.transactions?.[0]?.payment_date; }

// Accountants can narrow every table to one customer (name, email or payer ID): their disputes,
// invoice history (with when each was paid) and payments, to check a claim like "you charged me for two, I ordered one".
const CUSTOMER_FIELDS = {
  disputes: (d) => [buyerOf(d).name, buyerOf(d).email, buyerOf(d).payer_id],
  invoices: (i) => [recipientName(i), recipientOf(i)],
  transactions: (t) => [payerName(t), t.payer_info?.email_address, t.payer_info?.payer_id],
};
function forCustomer(view, rows) {
  const q = (state.customerFilter || "").trim().toLowerCase();
  if (!q) return rows;
  return rows.filter((r) => CUSTOMER_FIELDS[view](r).some((f) => String(f || "").toLowerCase().includes(q)));
}
function customerNames(d) {
  const names = new Set([...d.disputes.map((x) => buyerOf(x).name), ...d.invoices.map(recipientName),
    ...(d.transactions || []).map(payerName)].filter(Boolean));
  return [...names].sort();
}

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
    head: ["Invoice", "Recipient", "Status", "Amount", "Amount due", "Due date", "Issued", ""],
    row: (i) => { const s = invoiceStatus(i);
      const canSend = state.user.role === "accountant" && i.status === "DRAFT";
      const canPay = state.user.role === "customer" && ["SENT", "UNPAID", "PARTIALLY_PAID"].includes(i.status);
      const canChase = state.user.role === "accountant" && ["SENT", "UNPAID", "PARTIALLY_PAID"].includes(i.status);
      const actions = `<div class="row-actions">
          <button class="btn btn-ghost btn-sm" data-inv="download" data-id="${esc(i.id)}" data-number="${esc(i.detail?.invoice_number || "invoice")}">⬇ PDF</button>
          ${canSend ? `<button class="btn btn-primary btn-sm" data-inv="send" data-id="${esc(i.id)}" data-to="${esc(recipientOf(i))}">Send</button>` : ""}
          ${canChase ? `<button class="btn btn-ghost btn-sm" data-inv="remind" data-id="${esc(i.id)}">Remind</button>
            <button class="btn btn-ghost btn-sm" data-inv="mark-paid" data-id="${esc(i.id)}">Mark paid</button>
            <button class="btn btn-ghost btn-sm" data-inv="cancel" data-id="${esc(i.id)}">Cancel</button>` : ""}
          ${canPay ? `<button class="btn btn-primary btn-sm" data-inv="pay" data-id="${esc(i.id)}" data-amount="${esc(i.due_amount?.value || "")}">Pay ${money(i.due_amount)}</button>` : ""}
        </div>`;
      const items = (i.items || []).map((it) => `${esc(it.name)} × ${esc(it.quantity)}`).join(", ");
      const paid = paidOn(i) ? `<div class="desc">paid ${esc(fmtDate(paidOn(i)))}</div>` : "";
      return [`<b>${esc(i.detail?.invoice_number)}</b><div class="desc">${items}</div><div class="desc mono">${esc(i.id)}</div>`,
        `${recipientName(i) ? `${esc(recipientName(i))}<div class="desc">${esc(recipientOf(i))}</div>` : esc(recipientOf(i))}`,
        `<span class="badge ${s.kind}">${esc(s.text)}</span>${paid}`, money(i.amount), money(i.due_amount),
        esc(i.detail?.payment_term?.due_date || "—"), esc(i.detail?.invoice_date), actions]; },
  },
  transactions: {
    label: "Transactions",
    head: ["Transaction", "Type", "Paid how", "Customer", "Amount", "Fee", "Date"],
    row: (t) => { const i = t.transaction_info, p = t.payer_info || {}; const neg = Number(i.transaction_amount.value) < 0;
      return [`<code>${esc(i.transaction_id)}</code>`, badge(i.transaction_event_code === "T1107" ? "refund" : "sale", i.transaction_event_code === "T1107" ? "declined" : "ok"),
        paidHow(t),
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
  $("#customer-filter-wrap").hidden = d.role !== "accountant";
  if (d.role === "accountant") $("#customer-names").innerHTML = customerNames(d).map((n) => `<option value="${esc(n)}">`).join("");
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
  renderTransactions();
}

// Totals follow the customer filter, so "Priya: 3 sales, 1 refund" is one glance.
function renderTransactions() {
  const all = state.tx.transactions, rows = forCustomer("transactions", all);
  let t = state.tx.totals;
  if (rows !== all) {
    const sum = (code) => rows.filter((r) => r.transaction_info.transaction_event_code === code)
      .reduce((s, r) => s + Number(r.transaction_info.transaction_amount.value), 0);
    const fees = rows.reduce((s, r) => s + Number(r.transaction_info.fee_amount?.value || 0), 0);
    t = { sales: sum("T0006"), refunds: sum("T1107"), fees, net: sum("T0006") + sum("T1107") + fees, count: rows.length };
  }
  $("#tx-totals").innerHTML = [
    ["Sales", `<span class="amount-pos">${money({ value: t.sales })}</span>`],
    ["Refunds", `<span class="amount-neg">${money({ value: t.refunds })}</span>`],
    ["Fees", money({ value: t.fees })],
    ["Net", money({ value: t.net })],
    ["Transactions", t.count],
  ].map(([l, v]) => `<span class="chip">${l}<b>${v}</b></span>`).join("");
  renderRows(DATA_VIEWS.transactions, rows, rows === all ? "No transactions in this period." : "No transactions for this customer in this period. Try ◀ for earlier months.");
  filterNote(rows.length, all.length);
}

function filterNote(shown, total) {
  $("#customer-filter-note").textContent = state.customerFilter?.trim() ? `Showing ${shown} of ${total}` : "";
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
  const view = DATA_VIEWS[state.dataTab], all = state.data[state.dataTab], rows = forCustomer(state.dataTab, all);
  renderRows(view, rows, rows === all ? "Nothing here." : "Nothing for this customer.");
  filterNote(rows.length, all.length);
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

function shipDue(i) {
  if (i.days_left === null || i.days_left === undefined) return "paid";
  if (i.days_left < 0) return `<span class="amount-neg">delivery date passed ${-i.days_left} day${i.days_left === -1 ? "" : "s"} ago</span>`;
  return `deliver by ${esc(fmtDate(i.expected_date))} (${i.days_left === 0 ? "today" : `${i.days_left} day${i.days_left === 1 ? "" : "s"} left`})`;
}

function whatsNewItem(i) {
  const about = i.reason ? `${esc(DISPUTE_REASONS[i.reason] || "Dispute")} · ${money(i.amount)}` : "";
  const quote = i.text ? `<div class="quote">“${esc(i.text.length > 140 ? i.text.slice(0, 140) + "…" : i.text)}”</div>` : "";
  const due = !i.action_needed ? "" : i.days_left === null ? " · action needed"
    : i.days_left < 0 ? ` · <span class="amount-neg">overdue by ${-i.days_left} day${i.days_left === -1 ? "" : "s"}</span>`
    : ` · respond by ${esc(new Date(i.due_date).toLocaleDateString(undefined, { day: "numeric", month: "short" }))} (${i.days_left === 0 ? "today" : `${i.days_left} day${i.days_left === 1 ? "" : "s"} left`})`;
  const act = i.seller_action || {};
  const shopDid = act.type === "replacement" ? `sent you a replacement${act.carrier ? ` (${esc(act.carrier)} ${esc(act.tracking_number || "")})` : ""}`
    : `refunded you ${money(act.amount)}`;
  const texts = {
    action_needed: [`🔴 Action needed · ${esc(i.with)}${due}`, "Open", "open"],
    photo_needed: [`📷 ${esc(i.with)} ${i.rejected ? "needs another photo" : "asked for a photo"} to arrange your replacement${i.text ? `: ${esc(i.text)}` : ""}`, "Add photo", "open"],
    photos_to_review: [`📷 ${esc(i.with)} sent photos · check them to arrange the replacement`, "Review", "open"],
    confirm_resolution: [`✅ ${esc(i.with)} ${shopDid} · ${timeAgo(i.time)}. If you're happy, mark the case resolved; if not, reply in the case.`, "Open", "open"],
    new_message: [`📬 New message from ${esc(i.with)} · ${timeAgo(i.time)}${due}`, "Open", "open"],
    case_closed: [`✅ ${esc(i.with)} closed their case · ${esc(i.outcome)}${i.amount_refunded ? ` (${money(i.amount_refunded)})` : ""} · ${timeAgo(i.time)}`, "View", "open"],
    refund_requested: [i.refund_request?.wants === "replacement"
      ? `🔁 ${esc(i.with)} asked for a replacement · ${timeAgo(i.time)}${due}`
      : `💸 ${esc(i.with)} asked for a refund of ${money(i.refund_request?.amount || i.amount)} · ${timeAgo(i.time)}${due}`, "Resolve", "open"],
    needs_reply: [`✍️ ${esc(i.with)} wrote ${timeAgo(i.time)} · waiting for your reply${due}`, "Reply", "open"],
    no_reply_yet: [`⏳ You wrote ${timeAgo(i.time)} · no reply yet from ${esc(i.with)}`, "Send a reminder", "remind"],
    new_po: [`📥 New purchase order from ${esc(i.with)} · ${timeAgo(i.time)}${i.requested_date ? ` · needed by ${esc(fmtDate(i.requested_date))}` : ""}`, "Review", "po"],
    po_accepted: [`✅ Your purchase order was accepted · expected delivery ${esc(fmtDate(i.expected_date))}`, "View", "po"],
    po_rejected: [`❌ Your purchase order was declined`, "View", "po"],
    po_send_invoice: [`🧾 Send the invoice for ${esc(i.with)}'s order`, "Open", "po"],
    po_ship: [`📦 Ship ${esc(i.with)}'s order · ${shipDue(i)}`, "Ship", "po"],
    po_not_received: [`🔴 ${esc(i.with)} says the order hasn't arrived`, "Follow up", "po"],
    po_pay: [`💳 Pay for your order to start processing`, "Pay", "po"],
    po_shipped: [`🚚 Your order has shipped${i.tracking_number ? ` · ${esc(i.carrier || "")} ${esc(i.tracking_number)}` : ""} · expected ${esc(fmtDate(i.expected_date))}`, "Track", "po"],
    po_confirm: [`📦 Has your order arrived? It was expected ${esc(fmtDate(i.expected_date))}`, "Confirm", "po"],
  };
  if (i.kind === "refunded") {
    return `<div class="wn-item"><div><div class="what">💸 ${esc(i.with)} refunded you ${money(i.amount)} · ${timeAgo(i.time)}</div>
      ${i.text ? `<div class="quote">“${esc(i.text)}”</div>` : ""}<div class="quote">Refunds usually reach your account within a few days.</div></div></div>`;
  }
  if (i.kind === "invoice_overdue" || i.kind === "invoice_due") {
    const shop = state.user.role === "accountant";
    const when = i.days_left < 0 ? `<span class="amount-neg">overdue by ${-i.days_left} day${i.days_left === -1 ? "" : "s"}</span>`
      : i.days_left === 0 ? "due today" : `due in ${i.days_left} day${i.days_left === 1 ? "" : "s"}`;
    const what = shop ? `🧾 ${esc(i.with)}'s invoice is ${when}` : `🧾 Your invoice is ${when}`;
    return `<div class="wn-item"><div><div class="what">${what}</div><div class="quote">${esc(i.invoice_number || i.invoice_id)} · ${money(i.amount)} · due ${esc(fmtDate(i.due_date))}</div></div>
      ${shop ? `<div class="row-actions"><button class="btn btn-ghost" data-inv="remind" data-id="${esc(i.invoice_id)}">Remind</button>
        <button class="btn btn-ghost" data-inv="mark-paid" data-id="${esc(i.invoice_id)}">Mark paid</button></div>`
        : `<button class="btn btn-primary" data-wn="invoice">Pay</button>`}</div>`;
  }
  if (i.kind.startsWith("po") || i.kind === "new_po") {
    const [what, label] = texts[i.kind];
    const ref = i.customer_po_ref ? ` · your ref ${esc(i.customer_po_ref)}` : "";
    const detail = i.kind === "new_po" ? `${i.items} item${i.items === 1 ? "" : "s"}: ${esc(i.text)}`
      : i.kind === "po_rejected" ? `Reason: ${esc(i.text)}`
      : i.kind === "po_not_received" ? `“${esc(i.text)}”` : esc(i.text);
    return `<div class="wn-item"><div><div class="what">${what}</div><div class="quote">${esc(i.po_id)}${i.kind === "new_po" ? ref : ""}</div><div class="quote">${detail}</div></div>
      <button class="btn btn-ghost" data-wn="po" data-po="${esc(i.po_id)}">${label}</button></div>`;
  }
  const [what, label, action] = texts[i.kind];
  return `<div class="wn-item"><div><div class="what">${what}</div><div class="quote">${about}</div>${quote}</div>
    <button class="btn ${action === "remind" ? "btn-primary" : "btn-ghost"}" data-wn="${action}" data-dispute="${esc(i.dispute_id)}"
      data-days="${i.days || ""}">${label}</button></div>`;
}

// Builds the card, or updates it in place, so it always matches what's true right now.
// Called at login, when opening the Chat tab, and after every action that can change it.
async function loadWhatsNew() {
  if (!state.token) return;
  let res;
  try { res = await api("/api/whats-new"); } catch { return; }
  let card = $("#whats-new");
  if (!res.items.length) {
    if (card) card.remove();
    if (!$("#messages").children.length) $("#messages").append($("#chat-empty-template").content.cloneNode(true));
    return;
  }
  const html = `<h3>What's new</h3>${res.items.map(whatsNewItem).join("")}`;
  if (card) { card.innerHTML = html; return; }
  $("#chat-empty")?.remove();
  card = document.createElement("div");
  card.className = "whats-new";
  card.id = "whats-new";
  card.innerHTML = html;
  $("#messages").prepend(card);
  card.addEventListener("click", (e) => {
    const inv = e.target.closest("[data-inv]");
    if (inv) return invoiceAction(inv);
    const b = e.target.closest("[data-wn]");
    if (!b) return;
    if (b.dataset.wn === "open") return openDispute(b.dataset.dispute);
    if (b.dataset.wn === "po") { switchTab("orders"); return openPO(b.dataset.po); }
    if (b.dataset.wn === "invoice") { state.dataTab = "invoices"; return switchTab("data"); }
    const days = b.dataset.days ? ` I wrote ${b.dataset.days} days ago and haven't heard back.` : "";
    send(`Send a polite reminder on dispute ${b.dataset.dispute}.${days}`);
  });
}

// ---------------------------------------------------------------- invoices: download and send

async function downloadInvoice(id, number) {
  const res = await fetch(`/api/invoices/${encodeURIComponent(id)}/pdf`, { headers: { Authorization: `Bearer ${state.token}` } });
  if (res.status === 401) return logout("Your session has expired. Please log in again.");
  if (!res.ok) return toast((await res.json().catch(() => ({}))).detail || "Download failed.");
  const url = URL.createObjectURL(await res.blob());
  const a = Object.assign(document.createElement("a"), { href: url, download: `${number}.pdf` });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

// Accountants sort out unpaid invoices: remind, mark as paid (paid another way) or cancel. Second click confirms.
const INVOICE_ACTIONS = {
  remind: { ask: "Send reminder?", done: "Reminder sent to the customer." },
  "mark-paid": { ask: "Mark paid (bank transfer)?", done: "Marked as paid.", body: { method: "BANK_TRANSFER" } },
  cancel: { ask: "Cancel this invoice?", done: "Invoice cancelled." },
};
async function invoiceAction(btn) {
  const a = INVOICE_ACTIONS[btn.dataset.inv];
  if (!btn.dataset.armed) {
    btn.dataset.armed = "1"; const label = btn.textContent; btn.textContent = a.ask;
    setTimeout(() => { if (btn.isConnected && btn.dataset.armed) { delete btn.dataset.armed; btn.textContent = label; } }, 5000);
    return;
  }
  btn.disabled = true;
  try {
    await api(`/api/invoices/${encodeURIComponent(btn.dataset.id)}/${btn.dataset.inv}`, { method: "POST", body: JSON.stringify(a.body || {}) });
    toast(a.done);
    loadData(); loadWhatsNew();
  } catch (err) { if (err.message !== "unauthorized") toast(err.message); btn.disabled = false; }
}

async function payInvoice(btn) {
  // a second click confirms, like Send
  if (!btn.dataset.armed) {
    btn.dataset.armed = "1";
    btn.textContent = `Confirm payment of ${money({ value: btn.dataset.amount })}?`;
    setTimeout(() => { if (btn.isConnected && btn.dataset.armed) { delete btn.dataset.armed; btn.textContent = `Pay ${money({ value: btn.dataset.amount })}`; } }, 5000);
    return;
  }
  btn.disabled = true;
  try {
    await api(`/api/invoices/${encodeURIComponent(btn.dataset.id)}/pay`, { method: "POST" });
    toast("Paid with PayPal. Thank you!");
    loadData(); loadWhatsNew();
  } catch (err) { if (err.message !== "unauthorized") { toast(err.message); btn.disabled = false; } }
}

async function sendInvoice(btn) {
  // a second click within 5 seconds confirms (no browser pop-ups)
  if (!btn.dataset.armed) {
    if (!btn.dataset.to || btn.dataset.to === "—") return toast("Add a recipient email before sending this invoice.");
    btn.dataset.armed = "1";
    btn.textContent = `Send to ${btn.dataset.to}?`;
    setTimeout(() => { if (btn.isConnected && btn.dataset.armed) { delete btn.dataset.armed; btn.textContent = "Send"; } }, 5000);
    return;
  }
  btn.disabled = true;
  try {
    await api(`/api/invoices/${encodeURIComponent(btn.dataset.id)}/send`, { method: "POST" });
    toast(`Invoice sent to ${btn.dataset.to}.`);
    loadData(); loadWhatsNew();
  } catch (err) {
    if (err.message !== "unauthorized") { toast(err.message); btn.disabled = false; delete btn.dataset.armed; btn.textContent = "Send"; }
  }
}

// ---------------------------------------------------------------- purchase orders

const fmtDate = (d) => (d ? new Date(`${d}T00:00:00`).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" }) : "—");
const todayIso = () => new Date(Date.now() - new Date().getTimezoneOffset() * 60000).toISOString().slice(0, 10);
let po = null;          // the PO open in the panel (or a new, unsaved one)
let poDocUrl = null;    // object URL of its document preview

function poStatus(p) {
  const shop = state.user.role === "accountant";
  return {
    draft: { text: "Draft · not sent", kind: "neutral" },
    submitted: shop ? { text: "Needs review", kind: "error" } : { text: "Sent · waiting for the shop", kind: "waiting" },
    accepted: shop ? { text: "Accepted · send the invoice", kind: "waiting" } : { text: "Accepted · invoice coming", kind: "ok" },
    invoiced: shop ? { text: "Invoiced · awaiting payment", kind: "waiting" } : { text: "Pay the invoice", kind: "error" },
    paid: shop ? { text: "Paid · ship it", kind: "error" } : { text: "Paid · processing", kind: "ok" },
    shipped: { text: "Shipped", kind: "ok" },
    delivered: { text: "Delivered", kind: "ok" },
    not_received: shop ? { text: "Not received · follow up", kind: "error" } : { text: "Reported not received", kind: "waiting" },
    rejected: { text: "Declined", kind: "neutral" },
  }[p.status];
}

// The order's journey, for the PO panel: each step with its date, done or not yet.
function poTimeline(p) {
  const steps = [  // [label, date shown, done?]
    ["Sent to the shop", p.created_at, p.status !== "draft"],
    ["Accepted", null, Boolean(p.invoice_id)],
    ["Paid", p.paid_at, Boolean(p.paid_at)],
    ["Shipped", p.shipped_at, Boolean(p.shipped_at)],
    [p.status === "not_received" ? "Reported not received" : "Delivered", p.delivered_at || p.not_received_at, Boolean(p.delivered_at || p.not_received_at)],
  ];
  if (p.status === "rejected") return "";
  return `<ol class="timeline">${steps.map(([label, when, done]) =>
    `<li class="${done ? "done" : ""}"><span>${esc(label)}</span><small>${done && when ? esc(fmtDate(String(when).slice(0, 10))) : ""}</small></li>`).join("")}</ol>`;
}

// One line telling the viewer where the order stands and what (if anything) they need to do.
function poStageNote(p, shop) {
  const days = p.expected_date ? Math.round((new Date(`${p.expected_date}T00:00:00`) - new Date(`${todayIso()}T00:00:00`)) / 86400000) : null;
  const when = days === null ? "" : days > 0 ? ` (${days} day${days === 1 ? "" : "s"} left)` : days === 0 ? " (today)" : ` (${-days} day${days === -1 ? "" : "s"} ago)`;
  const good = (t) => `<div class="unclear" style="background:var(--ok-soft);color:var(--ok)">${t}</div>`;
  const todo = (t) => `<div class="unclear">${t}</div>`;
  const notes = shop ? {
    accepted: todo("A draft invoice was created. Send it so the customer can pay."),
    invoiced: good("Invoice sent. Waiting for the customer to pay."),
    paid: todo(`Paid. Ship it by <b>${fmtDate(p.expected_date)}</b>${when} and add the tracking number.`),
    shipped: good(`On its way. The customer confirms when it arrives.`),
    delivered: good(`Delivered: the customer confirmed on ${fmtDate(String(p.delivered_at).slice(0, 10))}.`),
    not_received: todo(`The customer says it hasn't arrived${p.not_received_note ? `: “${esc(p.not_received_note)}”` : "."} Check with the carrier, then ship again (or refund from the chat).`),
  } : {
    accepted: good(`Accepted. Your invoice is on its way; expected delivery <b>${fmtDate(p.expected_date)}</b>.`),
    invoiced: todo(`Please pay the invoice to start processing. Expected delivery <b>${fmtDate(p.expected_date)}</b>.`),
    paid: days !== null && days < 0 ? todo(`Expected ${fmtDate(p.expected_date)}${when}. Has it arrived?`)
      : good(`Paid, thank you. Your order is being prepared; expected delivery <b>${fmtDate(p.expected_date)}</b>${when}.`),
    shipped: days !== null && days <= 0 ? todo(`It should have arrived by now. Did you get it?`)
      : good(`On its way. Expected <b>${fmtDate(p.expected_date)}</b>${when}.`),
    delivered: good("Delivered. Thanks for your order!"),
    not_received: todo("You reported it hasn't arrived. The shop has been told and will follow up."),
  };
  return notes[p.status] || "";
}

async function poAction(path, body, message) {
  try {
    const res = await api(`/api/pos/${encodeURIComponent(po.po_id)}/${path}`, { method: "POST", body: JSON.stringify(body || {}) });
    toast(message);
    showPO(res); loadOrders(); loadWhatsNew();
  } catch (err) { if (err.message !== "unauthorized") toast(err.message); }
}

async function poPay() {
  try {
    await api(`/api/invoices/${encodeURIComponent(po.invoice_id)}/pay`, { method: "POST" });
    toast("Paid with PayPal. Thank you!");
    openPO(po.po_id); loadOrders(); loadWhatsNew();
  } catch (err) { if (err.message !== "unauthorized") toast(err.message); }
}

async function poSendInvoice() {
  try {
    await api(`/api/invoices/${encodeURIComponent(po.invoice_id)}/send`, { method: "POST" });
    toast(`Invoice sent to ${po.customer?.email || "the customer"}.`);
    openPO(po.po_id); loadOrders(); loadWhatsNew();
  } catch (err) { if (err.message !== "unauthorized") toast(err.message); }
}

function poShip() {
  const carrier = $("#po-carrier").value.trim(), tracking = $("#po-tracking").value.trim();
  if (carrier.length < 2 || tracking.length < 3) return toast("Add the carrier and tracking number.");
  poAction("ship", { carrier, tracking_number: tracking }, `${po.po_id} marked as shipped.`);
}

function poMissing() {
  const wrap = $("#po-missing-wrap");
  if (wrap.hidden) { wrap.hidden = false; $("#po-missing-note").focus(); $("#po-missing").textContent = "Report it missing"; return; }
  poAction("not-received", { note: $("#po-missing-note").value.trim() || null }, "Reported. The shop will follow up.");
}

async function loadOrders() {
  $("#orders-table").innerHTML = `<tr><td class="empty-row">Loading…</td></tr>`;
  let res;
  try { res = await api("/api/pos"); } catch (err) { if (err.message !== "unauthorized") $("#orders-table").innerHTML = `<tr><td class="empty-row">${esc(err.message)}</td></tr>`; return; }
  const shop = state.user.role === "accountant";
  const head = shop ? ["PO", "Customer", "Items", "Needed by", "Status", "Delivery", "Invoice"] : ["PO", "Your ref", "Items", "Needed by", "Status", "Delivery", "Invoice"];
  const rows = res.items.map((p) => {
    const s = poStatus(p);
    const items = p.items.map((i) => `${i.quantity} × ${esc(i.name)}`).join(", ") || "—";
    const invoice = p.invoice_id ? `<button class="btn btn-ghost btn-sm" data-inv="download" data-id="${esc(p.invoice_id)}" data-number="${esc(p.po_id)}-invoice">⬇ PDF</button>` : "—";
    return `<tr class="clickable" data-po="${esc(p.po_id)}"><td><b>${esc(p.po_id)}</b>${p.has_document ? ` <span class="desc">📎</span>` : ""}</td>
      <td>${shop ? esc(p.customer?.name || "") : esc(p.customer_po_ref || "—")}</td><td><div class="desc">${items}</div></td>
      <td>${fmtDate(p.requested_date)}</td><td><span class="badge ${s.kind}">${esc(s.text)}</span></td>
      <td>${p.expected_date ? fmtDate(p.expected_date) : "—"}</td><td>${invoice}</td></tr>`;
  }).join("");
  $("#orders-table").innerHTML = `<tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr>` +
    (rows || `<tr><td class="empty-row" colspan="${head.length}">${shop ? "No purchase orders yet." : "No purchase orders yet. Upload a photo of your PO or type one in."}</td></tr>`);
}

async function apiUpload(path, formData) {
  const res = await fetch(path, { method: "POST", body: formData, headers: { Authorization: `Bearer ${state.token}` } });
  if (res.status === 401) { logout("Your session has expired. Please log in again."); throw new Error("unauthorized"); }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Upload failed (${res.status})`);
  return body;
}

async function uploadPO(file) {
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  // reading a photo with the AI can take a while: say so, and keep the buttons disabled until it's done
  const label = $("#po-upload-label"), typeBtn = $("#po-type");
  const labelText = label.firstChild.textContent;  // the text before the hidden <input>
  label.classList.add("busy");
  label.firstChild.textContent = "⏳ Reading your PO… (up to a minute)";
  typeBtn.disabled = true;
  $("#po-file").disabled = true;
  try {
    const res = await apiUpload("/api/pos/read", fd);
    showPO(res.po, { unclear: res.unclear, readOk: res.read_ok });
    loadOrders();
  } catch (err) { if (err.message !== "unauthorized") toast(err.message); }
  finally {
    label.classList.remove("busy");
    label.firstChild.textContent = labelText;
    typeBtn.disabled = false;
    $("#po-file").disabled = false;
    $("#po-file").value = "";
  }
}

async function openPO(id) {
  try { showPO(await api(`/api/pos/${encodeURIComponent(id)}`)); } catch (err) { if (err.message !== "unauthorized") toast(err.message); }
}

function itemRow(i = {}, withPrice = true) {
  return `<tr>
    <td><input data-f="name" value="${esc(i.name || "")}" placeholder="Item"></td>
    <td class="qty"><input data-f="quantity" type="number" min="1" step="1" value="${esc(i.quantity || 1)}"></td>
    ${withPrice ? `<td class="price"><input data-f="unit_price" class="num" inputmode="decimal" value="${esc(i.unit_price || "")}" placeholder="${state.user.role === "accountant" ? "0.00" : "optional"}"></td>` : ""}
    <td class="del"><button type="button" class="btn btn-ghost btn-icon" data-del title="Remove">✕</button></td></tr>`;
}

function readItems() {
  return $$("#po-items tbody tr").map((tr) => {
    const get = (f) => tr.querySelector(`[data-f="${f}"]`)?.value.trim() ?? "";
    return { name: get("name"), quantity: parseInt(get("quantity") || "0", 10), unit_price: get("unit_price") || null };
  }).filter((i) => i.name);
}

function updateTotal() {
  const items = readItems();
  const priced = items.length && items.every((i) => i.unit_price && !isNaN(Number(i.unit_price)));
  const el = $("#po-total");
  if (el) el.textContent = priced ? `Total ${money({ value: items.reduce((s, i) => s + Number(i.unit_price) * i.quantity, 0) })}` : "";
}

async function showDocument(p) {
  if (poDocUrl) { URL.revokeObjectURL(poDocUrl); poDocUrl = null; }
  const box = $("#po-doc");
  if (!box) return;
  try {
    const res = await fetch(`/api/pos/${encodeURIComponent(p.po_id)}/document`, { headers: { Authorization: `Bearer ${state.token}` } });
    if (!res.ok) throw new Error();
    poDocUrl = URL.createObjectURL(await res.blob());
    box.innerHTML = p.document_type === "application/pdf" ? `<iframe src="${poDocUrl}" title="Purchase order document"></iframe>` : `<img src="${poDocUrl}" alt="Purchase order document">`;
  } catch { box.innerHTML = `<p class="muted">Couldn't load the document.</p>`; }
}

function showPO(p, { unclear = [], readOk = true } = {}) {
  po = p;
  const shop = state.user.role === "accountant";
  const s = p.status ? poStatus(p) : { text: "New", kind: "neutral" };
  const editable = (!shop && (!p.status || p.status === "draft")) || (shop && p.status === "submitted");
  $("#po-title").textContent = p.po_id ? `Purchase order ${p.po_id}` : "New purchase order";
  $("#po-sub").innerHTML = `<span class="badge ${s.kind}">${esc(s.text)}</span>` +
    (shop && p.customer ? `<span>from ${esc(p.customer.name)} · ${esc(p.customer.email)}</span>` : "") +
    (p.customer_po_ref ? `<span>ref ${esc(p.customer_po_ref)}</span>` : "");

  const notes = [];
  if (!readOk) notes.push("The document couldn't be read automatically right now. Please fill in the details below.");
  if (unclear.length) notes.push(`Please double-check: <ul>${unclear.map((u) => `<li>${esc(u)}</li>`).join("")}</ul>`);
  const notesHtml = notes.length ? `<div class="unclear">${notes.join("<br>")}</div>` : "";

  let form;
  if (editable && !shop) {
    form = `${notesHtml}
      <div class="grid2">
        <label>Your PO number<input id="po-ref" value="${esc(p.customer_po_ref || "")}" placeholder="optional"></label>
        <label>Needed by<input id="po-date" type="date" min="${todayIso()}" value="${esc(p.requested_date || "")}"></label>
      </div>
      <table class="po-items" id="po-items"><thead><tr><th>Item</th><th>Qty</th><th>Unit price</th><th></th></tr></thead>
        <tbody>${(p.items?.length ? p.items : [{}]).map((i) => itemRow(i)).join("")}</tbody></table>
      <button type="button" class="btn btn-ghost" id="po-add">＋ Add item</button>
      <label>Notes for the shop<textarea id="po-notes" rows="3" placeholder="Delivery address, contact, anything else">${esc(p.notes || "")}</textarea></label>`;
  } else if (editable && shop) {
    form = `<div class="grid2">
        <div><div class="desc">Customer's ref</div><b>${esc(p.customer_po_ref || "—")}</b></div>
        <div><div class="desc">Needed by</div><b>${fmtDate(p.requested_date)}</b></div>
      </div>
      ${p.notes ? `<div><div class="desc">Customer's notes</div>${esc(p.notes)}</div>` : ""}
      <table class="po-items" id="po-items"><thead><tr><th>Item</th><th>Qty</th><th>Unit price</th><th></th></tr></thead>
        <tbody>${p.items.map((i) => itemRow(i)).join("")}</tbody></table>
      <button type="button" class="btn btn-ghost" id="po-add">＋ Add item</button>
      <div class="grid2">
        <label>Expected delivery<input id="po-expected" type="date" min="${todayIso()}" value="${esc(p.requested_date && p.requested_date >= todayIso() ? p.requested_date : "")}"></label>
        <label>Note on the invoice<input id="po-note" placeholder="optional"></label>
      </div>
      <label id="po-reject-wrap" hidden>Reason for declining<input id="po-reason" placeholder="e.g. out of stock until November"></label>`;
  } else {
    form = `<div class="grid2">
        <div><div class="desc">${shop ? "Customer's ref" : "Your PO number"}</div><b>${esc(p.customer_po_ref || "—")}</b></div>
        <div><div class="desc">Needed by</div><b>${fmtDate(p.requested_date)}</b></div>
      </div>
      <ul class="readonly-list">${p.items.map((i) => `<li>${i.quantity} × ${esc(i.name)}${i.unit_price ? ` · ${money({ value: i.unit_price })} each` : ""}</li>`).join("")}</ul>
      ${p.total ? `<div><b>Total ${money({ value: p.total })}</b></div>` : ""}
      ${p.notes ? `<div><div class="desc">Notes</div>${esc(p.notes)}</div>` : ""}
      ${poTimeline(p)}
      ${p.expected_date ? `<div><div class="desc">Expected delivery</div><b>${fmtDate(p.expected_date)}</b></div>` : ""}
      ${p.tracking_number ? `<div class="tracking"><div class="desc">Tracking</div><b>${esc(p.carrier || "")} · ${esc(p.tracking_number)}</b></div>` : ""}
      ${poStageNote(p, shop)}
      ${shop && ["paid", "not_received"].includes(p.status) ? `<div class="grid2">
          <label>Carrier<input id="po-carrier" placeholder="e.g. FedEx, DHL, Blue Dart" value="${esc(p.carrier || "")}"></label>
          <label>Tracking number<input id="po-tracking" placeholder="e.g. 1Z999AA10123456784"></label></div>` : ""}
      ${!shop && ["shipped", "paid"].includes(p.status) ? `<label id="po-missing-wrap" hidden>What happened? (optional)<input id="po-missing-note" placeholder="e.g. tracking says delivered but nothing arrived"></label>` : ""}
      ${p.status === "rejected" ? `<div class="unclear">Declined: ${esc(p.reject_reason || "")}</div>` : ""}`;
  }

  $("#po-body").className = `po-body ${p.has_document ? "" : "single"}`;
  $("#po-body").innerHTML = `${p.has_document ? `<div class="po-doc" id="po-doc"><p class="muted">Loading document…</p></div>` : ""}<div class="po-form">${form}</div>`;

  let foot = "";
  if (editable && !shop) foot = `<span class="po-total" id="po-total"></span><button class="btn btn-ghost" id="po-save">Save draft</button><button class="btn btn-primary" id="po-submit">Send to the shop</button>`;
  if (editable && shop) foot = `<span class="po-total" id="po-total"></span><button class="btn btn-ghost" id="po-reject">Decline</button><button class="btn btn-primary" id="po-accept">Accept &amp; create invoice</button>`;
  if (!editable) {
    const pdf = p.invoice_id ? `<button class="btn btn-ghost" data-inv="download" data-id="${esc(p.invoice_id)}" data-number="${esc(p.po_id)}-invoice">⬇ Invoice PDF</button>` : "";
    const next = shop ? {
      accepted: `<button class="btn btn-primary" id="po-send-invoice">Send invoice to ${esc(p.customer?.email || "customer")}</button>`,
      paid: `<button class="btn btn-primary" id="po-ship">Mark as shipped</button>`,
      not_received: `<button class="btn btn-primary" id="po-ship">Ship again</button>`,
    }[p.status] : {
      invoiced: `<button class="btn btn-primary" id="po-pay">Pay ${p.total ? money({ value: p.total }) : ""} with PayPal</button>`,
      shipped: `<button class="btn btn-ghost" id="po-missing">Not received</button><button class="btn btn-primary" id="po-delivered">Yes, it arrived</button>`,
      paid: p.expected_date && p.expected_date < todayIso()
        ? `<button class="btn btn-ghost" id="po-missing">Not received</button><button class="btn btn-primary" id="po-delivered">Yes, it arrived</button>` : "",
    }[p.status];
    foot = `${pdf}${next || ""}`;
  }
  $("#po-foot").innerHTML = foot;
  $("#po-foot").hidden = !foot;

  $("#po-drawer").hidden = $("#po-backdrop").hidden = false;
  if (p.has_document) showDocument(p);
  updateTotal();
}

function closePO() {
  $("#po-drawer").hidden = $("#po-backdrop").hidden = true;
  if (poDocUrl) { URL.revokeObjectURL(poDocUrl); poDocUrl = null; }
  po = null;
}

function customerDraft() {
  return { items: readItems(), customer_po_ref: $("#po-ref").value.trim() || null, requested_date: $("#po-date").value || null, notes: $("#po-notes").value.trim() || null };
}

async function savePO(send) {
  const body = customerDraft();
  if (send && !body.items.length) return toast("Add at least one item.");
  try {
    let saved = po.po_id ? await api(`/api/pos/${encodeURIComponent(po.po_id)}`, { method: "PUT", body: JSON.stringify(body) })
                         : await api("/api/pos", { method: "POST", body: JSON.stringify(body) });
    if (send) saved = await api(`/api/pos/${encodeURIComponent(saved.po_id)}/submit`, { method: "POST" });
    toast(send ? `${saved.po_id} sent to the shop.` : `${saved.po_id} saved as a draft.`);
    closePO(); loadOrders(); loadWhatsNew();
  } catch (err) { if (err.message !== "unauthorized") toast(err.message); }
}

async function decidePO(accept) {
  if (accept) {
    const items = readItems();
    // show exactly what's missing: price fields and the date turn red until filled in
    const missingPrices = $$("#po-items tbody tr").filter((tr) => tr.querySelector('[data-f="name"]').value.trim()
      && !tr.querySelector('[data-f="unit_price"]').value.trim());
    $$("#po-items input.missing, #po-expected.missing").forEach((el) => el.classList.remove("missing"));
    missingPrices.forEach((tr) => tr.querySelector('[data-f="unit_price"]').classList.add("missing"));
    if (!items.length || missingPrices.length) {
      missingPrices[0]?.querySelector('[data-f="unit_price"]').focus();
      return toast(`Add a unit price for ${missingPrices.length === 1 ? `“${missingPrices[0].querySelector('[data-f="name"]').value.trim()}”` : `${missingPrices.length} items`} (the customer didn't include one).`);
    }
    const expected = $("#po-expected").value;
    if (!expected) { $("#po-expected").classList.add("missing"); $("#po-expected").focus(); return toast("Pick an expected delivery date."); }
    try {
      const res = await api(`/api/pos/${encodeURIComponent(po.po_id)}/accept`, { method: "POST", body: JSON.stringify({ items, expected_date: expected, note: $("#po-note").value.trim() || null }) });
      toast(`${res.po_id} accepted. A draft invoice was created (see PayMind data → Invoices).`);
      closePO(); loadOrders(); loadWhatsNew();
    } catch (err) { if (err.message !== "unauthorized") toast(err.message); }
    return;
  }
  const wrap = $("#po-reject-wrap");
  if (wrap.hidden) { wrap.hidden = false; $("#po-reason").focus(); $("#po-reject").textContent = "Confirm decline"; return; }
  const reason = $("#po-reason").value.trim();
  if (reason.length < 3) return toast("Add a short reason for the customer.");
  try {
    const res = await api(`/api/pos/${encodeURIComponent(po.po_id)}/reject`, { method: "POST", body: JSON.stringify({ reason }) });
    toast(`${res.po_id} declined.`); closePO(); loadOrders(); loadWhatsNew();
  } catch (err) { if (err.message !== "unauthorized") toast(err.message); }
}

// ---------------------------------------------------------------- dispute conversation

let openDisputeId = null;

let threadState = null;  // the open dispute (for the resolve box)

function resolveChoice() { return document.querySelector('input[name="resolve"]:checked').value; }

function updateResolveBox() {
  const approved = threadState?.dispute?.evidence?.status === "approved";
  const rep = document.querySelector('input[name="resolve"][value="replacement"]');
  rep.disabled = !approved;
  rep.closest("label").querySelector("small").textContent = approved
    ? "Free: ship a new one; the customer gets the tracking ID"
    : "Ask for photos first (📷 in the case) and approve them";
  if (!approved && rep.checked) document.querySelector('input[name="resolve"][value="refund"]').checked = true;
  const choice = resolveChoice(), amount = threadState?.dispute?.dispute_amount;
  $$(".resolve-fields [data-for]").forEach((el) => { el.hidden = el.dataset.for !== choice; });
  const partial = choice === "refund" && $("#resolve-amount").value.trim();
  $("#resolve-confirm").textContent = { refund: `Refund ${partial ? money({ value: partial }) : money(amount)}`, replacement: "Send replacement" }[choice];
  $("#resolve-confirm").dataset.armed = "";
}

function openResolve() {
  $("#resolve-box").hidden = false;
  $("#resolve-open").hidden = true;
  $("#thread-form").hidden = true;
  $("#resolve-refund-text").textContent = `Give ${threadState.dispute.disputed_transactions?.[0]?.buyer?.name || "the customer"} their ${money(threadState.dispute.dispute_amount)} back, or part of it`;
  const wants = threadState.dispute.refund_request?.wants;  // start on what the customer asked for
  if (wants) document.querySelector(`input[name="resolve"][value="${wants === "replacement" ? "replacement" : "refund"}"]`).checked = true;
  updateResolveBox();
}

function closeResolve() {
  $("#resolve-box").hidden = true;
  if (threadState) renderThread(threadState);
}

async function confirmResolve() {
  const btn = $("#resolve-confirm"), choice = resolveChoice();
  const body = { action: choice, note: $("#resolve-note").value.trim() || null };
  if (choice === "refund" && $("#resolve-amount").value.trim()) {
    body.amount = $("#resolve-amount").value.trim();
    if (isNaN(Number(body.amount)) || Number(body.amount) <= 0) { $("#resolve-amount").focus(); return toast("Enter a valid amount, or leave it empty for the full refund."); }
  }
  if (choice === "replacement") {
    body.carrier = $("#resolve-carrier").value.trim();
    body.tracking_number = $("#resolve-tracking").value.trim();
    if (body.carrier.length < 2) { $("#resolve-carrier").focus(); return toast("Add the carrier."); }
    if (body.tracking_number.length < 3) { $("#resolve-tracking").focus(); return toast("Add the tracking number."); }
  }
  if (!btn.dataset.armed) {  // second click confirms: money and case outcomes are hard to undo
    btn.dataset.armed = "1";
    btn.textContent = choice === "refund" ? `Confirm refund of ${money(body.amount ? { value: body.amount } : threadState.dispute.dispute_amount)}?`
      : "Confirm: send replacement?";
    return;
  }
  btn.disabled = true;
  try {
    const res = await api(`/api/disputes/${encodeURIComponent(openDisputeId)}/resolve`, { method: "POST", body: JSON.stringify(body) });
    toast({ refund: "Refunded. The customer will confirm and close the case.",
      replacement: "Replacement recorded. The customer has the tracking number and will confirm." }[choice]);
    ["#resolve-amount", "#resolve-note", "#resolve-carrier", "#resolve-tracking"].forEach((s) => { $(s).value = ""; });
    $("#resolve-box").hidden = true;
    renderThread(res);
    loadData(); loadWhatsNew();
  } catch (err) { if (err.message !== "unauthorized") toast(err.message); updateResolveBox(); }
  finally { btn.disabled = false; }
}

function renderThread(res) {
  threadState = res;
  const shopView = state.user.role === "accountant";
  $("#resolve-open").hidden = !(shopView && res.can_resolve && $("#resolve-box").hidden);
  $("#case-close").hidden = shopView || !res.can_reply;
  $("#case-close").dataset.armed = ""; $("#case-close").textContent = "✅ Mark as resolved";
  const d = res.dispute, s = disputeStatus(d), mine = state.user.role === "accountant" ? "SELLER" : "BUYER";
  $("#drawer-title").textContent = `${disputeReason(d)} · ${money(d.dispute_amount)}`;
  $("#drawer-sub").innerHTML = `<span class="mono">${esc(d.dispute_id)}</span><span class="badge ${s.kind}">${esc(s.text)}</span>` +
    (state.user.role === "accountant" ? `<span>with ${esc(buyerOf(d).name || "customer")}</span>` : "");
  const rr = d.refund_request;
  const asked = rr?.wants === "replacement" ? "for a <b>replacement</b>" : `for a refund of <b>${rr ? money(rr.amount) : ""}</b>`;
  const requestLine = rr ? `<div class="refund-request">${rr.wants === "replacement" ? "🔁" : "💸"} ${shopView ? `${esc(buyerOf(d).name || "The customer")} asked` : "You asked"} ${asked} · ${shortDate(rr.time)}${shopView ? " · use <b>Resolve…</b>" : " · waiting for the shop"}</div>` : "";
  const o = d.dispute_outcome || {};
  const ended = { RESOLVED_WITH_REPLACEMENT: `🔁 Replacement sent · ${esc(o.carrier || "")} ${esc(o.tracking_number || "")}`,
    RESOLVED_BUYER_FAVOUR: `💸 Refunded ${o.amount_refunded ? money(o.amount_refunded) : ""}`,
    CANCELED_BY_BUYER: "✅ Marked resolved by the customer" }[o.outcome_code];
  const outcome = d.status === "RESOLVED" && ended ? `<div class="refund-request">Case closed · ${ended}</div>` : "";
  const sa = res.seller_action && d.status !== "RESOLVED" ? res.seller_action : null;
  const did = sa ? (sa.type === "replacement" ? `sent a replacement · ${esc(sa.carrier || "")} <span class="mono">${esc(sa.tracking_number || "")}</span>`
    : `refunded <b>${money(sa.amount)}</b>`) : "";
  const offerHint = sa ? `<div class="refund-request">${sa.type === "replacement" ? "🔁" : "💸"} ${shopView ? "You" : "The shop"} ${did} · ${shortDate(sa.time)}.
    ${shopView ? "Waiting for the customer to confirm and close the case." : "If you're happy, click <b>✅ Mark as resolved</b>; if not, reply here."}</div>` : "";
  const evidenceBox = evidenceBanner(d, shopView, res.can_reply);
  const photos = (res.photos || []).length ? `<div class="photo-strip">${res.photos.map((ph) =>
    `<a class="photo" data-photo="${esc(ph.photo_id)}" title="${ph.by === mine ? "You" : "They"} · ${shortDate(ph.time)}"><span class="muted small">Loading…</span></a>`).join("")}</div>` : "";
  $("#thread").innerHTML = purchaseBox(res.purchase, shopView && res.can_reply) + requestLine + offerHint + outcome + evidenceBox + photos + res.messages.map((m) => `
    <div class="tmsg ${m.from === mine ? "mine" : ""}">
      <div class="who">${esc(m.from === mine ? "You" : m.name)} · ${shortDate(m.time)}</div>
      <div class="text">${esc(m.text.replace(/\\r?\\n/g, "\n"))}</div>
    </div>`).join("") || `<p class="muted">No messages yet.</p>`;
  $("#thread").scrollTop = $("#thread").scrollHeight;
  loadPhotos(d.dispute_id);
  $("#thread-form").hidden = !res.can_reply || !$("#resolve-box").hidden;
  $("#thread-closed").hidden = res.can_reply;
}

const SHIP_STATUS = { SHIPPED: "shipped · in transit", ON_HOLD: "on hold", DELIVERED: "delivered", CANCELLED: "shipment cancelled" };
const PAID_HOW = { invoice: "🧾 Invoice", "in store": "🏬 In store", "online store": "🌐 Online store" };

// What the case is about, and where it is: the items bought, how and when they were paid, and the
// latest shipment (the shop can record one here).
function purchaseBox(p, canTrack) {
  if (!p) return "";
  const items = p.items.length ? esc(p.items.join(", ")) : "Purchase";
  const refunded = Number(p.refunded) > 0 ? ` · ${money({ value: p.refunded })} refunded` : "";
  const ship = p.shipments[0];
  const shipLine = ship
    ? `📦 ${esc(ship.carrier)} <span class="mono">${esc(ship.tracking_number)}</span> · ${esc(SHIP_STATUS[ship.status] || ship.status)} · shipped ${esc(fmtDate(ship.shipment_date))}`
    : p.paid_how === "in store" ? "🏬 Collected in store" : "📦 No shipment recorded";
  const form = canTrack && p.paid_how !== "in store" ? `
    <details class="track-form"><summary>${ship ? "Update tracking" : "Add tracking"}</summary>
      <div class="track-fields">
        <input id="track-carrier" placeholder="Carrier, e.g. Blue Dart" value="${esc(ship?.carrier || "")}">
        <input id="track-number" placeholder="Tracking number" value="${esc(ship?.tracking_number || "")}">
        <select id="track-status">${Object.entries(SHIP_STATUS).map(([k, v]) => `<option value="${k}" ${ship?.status === k ? "selected" : ""}>${v}</option>`).join("")}</select>
        <button class="btn btn-primary btn-sm" type="button" id="track-save">Save</button>
      </div>
    </details>` : "";
  return `<div class="purchase-box"><div class="muted small">About this purchase</div>
    <div><b>${items}</b> · ${money({ value: p.amount })}${refunded}</div>
    <div class="muted small">${PAID_HOW[p.paid_how] || esc(p.paid_how)} · paid ${shortDate(p.date)} · payment <span class="mono">${esc(p.payment_id)}</span></div>
    <div class="ship-line">${shipLine}</div>${form}</div>`;
}

// Free replacements need photos first: the shop asks, the customer attaches (📎), the shop approves or asks again.
function evidenceBanner(d, shopView, open) {
  if (!open || d.seller_action) return "";
  const ev = d.evidence, name = esc(buyerOf(d).name || "The customer");
  if (!ev) {
    return shopView ? `<details class="track-form refund-request"><summary>📷 Ask for photos before a replacement</summary>
      <div class="track-fields"><input id="photos-note" placeholder="What should the photo show? e.g. the left earbud and its serial number">
      <button class="btn btn-primary btn-sm" type="button" id="photos-ask">Ask</button></div></details>` : "";
  }
  const note = ev.note ? `: ${esc(ev.note)}` : "";
  if (ev.status === "requested" || ev.status === "rejected") {
    const again = ev.status === "rejected" ? "needs another photo" : "asked for";
    return shopView ? `<div class="refund-request">📷 Waiting for ${name}'s photos${note}</div>`
      : `<div class="refund-request">📷 The shop ${again}${note}. Attach it with the <b>📎</b> button below.</div>`;
  }
  if (ev.status === "submitted") {
    return shopView ? `<div class="refund-request">📷 ${name} sent photos (below). Do they confirm the problem?
      <div class="track-fields"><button class="btn btn-primary btn-sm" type="button" id="photos-approve">✅ Yes, approve</button>
      <input id="photos-reject-note" placeholder="If not, what's missing?"><button class="btn btn-ghost btn-sm" type="button" id="photos-reject">❌ Ask for another</button></div></div>`
      : `<div class="refund-request">📷 Photos sent. The shop is checking them.</div>`;
  }
  return shopView ? `<div class="refund-request">✅ Photos approved. Send the free replacement from <b>Resolve…</b> with the tracking ID.</div>`
    : `<div class="refund-request">✅ The shop approved your photos. Your free replacement is being arranged; you'll get a tracking ID here.</div>`;
}

async function evidenceAction(kind) {
  const url = `/api/disputes/${encodeURIComponent(openDisputeId)}/${kind === "ask" ? "request-photos" : "review-photos"}`;
  const body = kind === "ask" ? { note: $("#photos-note")?.value.trim() || null }
    : { approved: kind === "approve", note: $("#photos-reject-note")?.value.trim() || null };
  try {
    renderThread(await api(url, { method: "POST", body: JSON.stringify(body) }));
    toast({ ask: "Asked the customer for photos.", approve: "Photos approved. You can now send the replacement.", reject: "Asked the customer for another photo." }[kind]);
    loadData(); loadWhatsNew();
  } catch (err) { if (err.message !== "unauthorized") toast(err.message); }
}

async function closeCase() {
  const btn = $("#case-close");
  if (!btn.dataset.armed) { btn.dataset.armed = "1"; btn.textContent = "Close this case? Click again"; return; }
  btn.disabled = true;
  try {
    renderThread(await api(`/api/disputes/${encodeURIComponent(openDisputeId)}/close`, { method: "POST", body: JSON.stringify({ message: "I'm happy with how this was sorted out, so I'm closing this case." }) }));
    toast("Case closed. The shop has been told.");
    loadData(); loadWhatsNew();
  } catch (err) { if (err.message !== "unauthorized") toast(err.message); }
  finally { btn.disabled = false; }
}

async function saveTracking() {
  const body = { carrier: $("#track-carrier").value.trim(), tracking_number: $("#track-number").value.trim(), status: $("#track-status").value };
  if (body.carrier.length < 2) { $("#track-carrier").focus(); return toast("Add the carrier."); }
  if (body.tracking_number.length < 3) { $("#track-number").focus(); return toast("Add the tracking number."); }
  try {
    renderThread(await api(`/api/disputes/${encodeURIComponent(openDisputeId)}/tracking`, { method: "POST", body: JSON.stringify(body) }));
    toast("Tracking saved. The customer can see it in the case.");
  } catch (err) { if (err.message !== "unauthorized") toast(err.message); }
}

// Photos need the login token, so each is fetched and shown from a local blob URL.
async function loadPhotos(disputeId) {
  for (const el of $$("#thread [data-photo]")) {
    try {
      const res = await fetch(`/api/disputes/${encodeURIComponent(disputeId)}/photos/${encodeURIComponent(el.dataset.photo)}`,
        { headers: { Authorization: `Bearer ${state.token}` } });
      if (!res.ok) throw new Error();
      const url = URL.createObjectURL(await res.blob());
      el.href = url; el.target = "_blank";
      el.innerHTML = `<img src="${url}" alt="Attached photo">`;
    } catch { el.innerHTML = `<span class="muted small">Photo unavailable</span>`; }
  }
}

async function attachPhoto(file) {
  if (!file || !openDisputeId) return;
  if (file.size > 8 * 1024 * 1024) return toast("The photo is larger than 8 MB.");
  const form = new FormData();
  form.append("file", file);
  toast("Uploading photo…");
  const res = await fetch(`/api/disputes/${encodeURIComponent(openDisputeId)}/photos`, { method: "POST", body: form,
    headers: { Authorization: `Bearer ${state.token}` } });
  if (res.status === 401) return logout("Your session has expired. Please log in again.");
  const body = await res.json().catch(() => ({}));
  if (!res.ok) return toast(body.detail || "Upload failed.");
  toast("Photo attached. The shop can see it.");
  renderThread(body);
}

async function openDispute(id) {
  openDisputeId = id;
  $("#resolve-box").hidden = true;
  $("#drawer").hidden = $("#drawer-backdrop").hidden = false;
  $("#thread").innerHTML = `<p class="muted">Loading…</p>`;
  try { renderThread(await api(`/api/disputes/${encodeURIComponent(id)}`)); $("#thread-input").focus(); }
  catch (err) { if (err.message !== "unauthorized") $("#thread").innerHTML = `<p class="muted">${esc(err.message)}</p>`; }
}

function closeDispute() {
  $("#drawer").hidden = $("#drawer-backdrop").hidden = true;
  $("#resolve-box").hidden = true;
  openDisputeId = null;
  loadData();  // refresh the "new" badges
  loadWhatsNew();
}

async function sendThreadMessage() {
  const text = $("#thread-input").value.trim();
  if (!text || !openDisputeId) return;
  $("#thread-send").disabled = true;
  try {
    renderThread(await api(`/api/disputes/${encodeURIComponent(openDisputeId)}/messages`, { method: "POST", body: JSON.stringify({ message: text }) }));
    $("#thread-input").value = "";
    loadWhatsNew();
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
  $("#customer-filter").addEventListener("input", (e) => {
    state.customerFilter = e.target.value;
    if (!state.data) return;
    if (state.dataTab === "transactions" && state.tx) renderTransactions(); else renderDataTable();
  });
  $("#audit-refresh").addEventListener("click", loadAudit);
  $("#data-table").addEventListener("click", (e) => {
    const inv = e.target.closest("[data-inv]");
    if (inv) return inv.dataset.inv === "download" ? downloadInvoice(inv.dataset.id, inv.dataset.number)
      : inv.dataset.inv === "pay" ? payInvoice(inv) : inv.dataset.inv === "send" ? sendInvoice(inv) : invoiceAction(inv);
    const row = e.target.closest("tr[data-dispute]");
    if (row) openDispute(row.dataset.dispute);
  });
  $("#po-file").addEventListener("change", (e) => uploadPO(e.target.files[0]));
  $("#po-type").addEventListener("click", () => showPO({ items: [] }));
  $("#po-close").addEventListener("click", closePO);
  $("#po-backdrop").addEventListener("click", closePO);
  $("#orders-table").addEventListener("click", (e) => {
    const inv = e.target.closest("[data-inv]");
    if (inv) return downloadInvoice(inv.dataset.id, inv.dataset.number);
    const row = e.target.closest("tr[data-po]");
    if (row) openPO(row.dataset.po);
  });
  $("#po-drawer").addEventListener("click", (e) => {
    const inv = e.target.closest("[data-inv]");
    if (inv) return downloadInvoice(inv.dataset.id, inv.dataset.number);
    if (e.target.closest("[data-del]")) { e.target.closest("tr").remove(); return updateTotal(); }
    if (e.target.id === "po-add") { $("#po-items tbody").insertAdjacentHTML("beforeend", itemRow()); return; }
    if (e.target.id === "po-save") return savePO(false);
    if (e.target.id === "po-submit") return savePO(true);
    if (e.target.id === "po-accept") return decidePO(true);
    if (e.target.id === "po-reject") return decidePO(false);
    if (e.target.id === "po-pay") return poPay();
    if (e.target.id === "po-send-invoice") return poSendInvoice();
    if (e.target.id === "po-ship") return poShip();
    if (e.target.id === "po-delivered") return poAction("delivered", {}, "Thanks for confirming!");
    if (e.target.id === "po-missing") return poMissing();
  });
  $("#po-drawer").addEventListener("input", (e) => { if (e.target.closest("#po-items")) updateTotal(); });
  $("#drawer-close").addEventListener("click", closeDispute);
  $("#thread-photo").addEventListener("change", (e) => { attachPhoto(e.target.files[0]); e.target.value = ""; });
  $("#thread").addEventListener("click", (e) => {
    if (e.target.id === "track-save") saveTracking();
    if (e.target.id === "photos-ask") evidenceAction("ask");
    if (e.target.id === "photos-approve") evidenceAction("approve");
    if (e.target.id === "photos-reject") evidenceAction("reject");
  });
  $("#resolve-open").addEventListener("click", openResolve);
  $("#case-close").addEventListener("click", closeCase);
  $("#resolve-cancel").addEventListener("click", closeResolve);
  $("#resolve-confirm").addEventListener("click", confirmResolve);
  $("#resolve-options").addEventListener("change", updateResolveBox);
  $("#resolve-amount").addEventListener("input", updateResolveBox);
  $("#drawer-backdrop").addEventListener("click", closeDispute);
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (!$("#po-drawer").hidden) closePO(); else if (!$("#drawer").hidden) closeDispute();
  });
  $("#thread-form").addEventListener("submit", (e) => { e.preventDefault(); sendThreadMessage(); });
  $("#thread-input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendThreadMessage(); } });
  $$("#tx-period button").forEach((b) => b.addEventListener("click", () => { state.txPeriod = b.dataset.period; state.txAnchor = new Date(); loadTransactions(); }));
  $("#tx-prev").addEventListener("click", () => shiftPeriod(-1));
  $("#tx-next").addEventListener("click", () => shiftPeriod(1));

  $("#login-view").hidden = false;  // always start at login
});
