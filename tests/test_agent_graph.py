"""Agent loop tests: a scripted fake LLM drives the real validator, scope checks,
executor, mock PayPal server and audit log. No Gemini quota used."""

import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from paymind.agent.executor import Executor, ToolRegistry
from paymind.agent.graph import MAX_STEPS, Deps, PayMindAgent
from paymind.app.database import AppDatabase
from paymind.mock_paypal.app import create_app

REGISTRY = ToolRegistry()


def usd(value):
    return {"currency_code": "USD", "value": value}


def call(name, args, cid="c1"):
    return {"name": name, "args": args, "id": cid}


def tool_results(messages):
    return [json.loads(m.content) if m.content.startswith(("{", "[")) else m.content
            for m in messages if isinstance(m, ToolMessage)]


class ScriptedLLM:
    """Each step is an AIMessage or a function(messages) -> AIMessage."""

    def __init__(self, steps):
        self.steps, self.calls, self.seen_tools = list(steps), 0, []

    def __call__(self, schemas):
        self.seen_tools.append([s["name"] for s in schemas])
        return self

    def invoke(self, messages):
        step = self.steps[min(self.calls, len(self.steps) - 1)]
        self.calls += 1
        return step(messages) if callable(step) else step


def fake_search(offer):
    def search(query, role=None, k=5, include_eval_only=True):
        names = [n for n in offer if role in REGISTRY.get(n)["allowed_roles"]] if role else offer
        return [(n, REGISTRY.get(n)["description"]) for n in names][:k]
    return search


@pytest.fixture
def world(tmp_path):
    mock = create_app(db_path=tmp_path / "mock.db", slow_seconds=0)
    executor = Executor(REGISTRY, base_url="http://testserver", client=TestClient(mock), sleep=lambda s: None)
    appdb = AppDatabase(tmp_path / "app.db")
    return mock, executor, appdb


def make_agent(world, llm, offer):
    mock, executor, appdb = world
    deps = Deps(llm_for=llm, search=fake_search(offer), executor=executor, registry=REGISTRY, appdb=appdb)
    return PayMindAgent(deps, InMemorySaver())


def completed_capture(mock):
    return next(c for c in mock.state.store.db.all("captures") if c["status"] == "COMPLETED")


def refunds_on(mock, capture_id):
    return [r for r in mock.state.store.db.all("refunds") if r["capture_id"] == capture_id]


# ---- reading ------------------------------------------------------------------------------

def test_read_question_end_to_end(world):
    llm = ScriptedLLM([
        AIMessage("", tool_calls=[call("list_disputes", {"dispute_state": "REQUIRED_ACTION"})]),
        lambda msgs: AIMessage(f"Found: {tool_results(msgs)[-1]['result']['items'][0]['dispute_id']}"),
    ])
    agent = make_agent(world, llm, ["list_disputes", "show_dispute_details"])
    reply = agent.send("u_asha", "s1", "Is there a dispute open from user_123?")
    assert reply.confirmation is None and reply.text.startswith("Found: PP-D-")
    assert "list_disputes" in llm.seen_tools[0] and "find_tools" in llm.seen_tools[0]
    log = world[2].recent_actions("u_asha")
    assert [(a["tool"], a["status"], a["http_status"]) for a in log] == [("list_disputes", "success", 200)]


# ---- writing: confirmation -------------------------------------------------------------------------

def refund_script(capture_id):
    return ScriptedLLM([
        AIMessage("", tool_calls=[call("refund_captured_payment", {"capture_id": capture_id, "amount": usd(5)})]),
        AIMessage("Done."),
    ])


def test_write_waits_for_yes_then_runs_once(world):
    mock = world[0]
    cap = completed_capture(mock)["id"]
    agent = make_agent(world, refund_script(cap), ["refund_captured_payment"])

    reply = agent.send("u_asha", "s1", f"refund 5 dollars on payment {cap}")
    assert reply.text is None
    q = reply.confirmation["question"]
    assert q.startswith(f"Refund captured payment for 5 USD (capture_id={cap})?") and "\n\nTo: " in q and "\nFor: " in q
    assert refunds_on(mock, cap) == []  # nothing happens before the yes

    reply = agent.answer("u_asha", "s1", approve=True)
    assert reply.text == "Done."
    assert len(refunds_on(mock, cap)) == 1
    entry = world[2].recent_actions("u_asha")[0]
    assert entry["status"] == "success" and entry["confirmed"] and entry["request_id"].startswith("pm-s1-")


def test_declined_write_does_nothing(world):
    mock = world[0]
    cap = completed_capture(mock)["id"]
    agent = make_agent(world, refund_script(cap), ["refund_captured_payment"])
    agent.send("u_asha", "s1", f"refund 5 dollars on payment {cap}")
    agent.answer("u_asha", "s1", approve=False)
    assert refunds_on(mock, cap) == []
    assert world[2].recent_actions("u_asha")[0]["status"] == "declined"


def test_resume_does_not_repeat_earlier_calls(world):
    """A read and a write in the same step: after the yes, each runs exactly once."""
    mock, _, appdb = world
    cap = completed_capture(mock)["id"]
    llm = ScriptedLLM([
        AIMessage("", tool_calls=[call("list_disputes", {}, "c1"),
                                  call("refund_captured_payment", {"capture_id": cap, "amount": usd(5)}, "c2")]),
        AIMessage("Done."),
    ])
    agent = make_agent(world, llm, ["list_disputes", "refund_captured_payment"])
    agent.send("u_asha", "s1", f"list disputes and refund 5 on {cap}")
    agent.answer("u_asha", "s1", approve=True)
    tools_run = [a["tool"] for a in appdb.recent_actions("u_asha")]
    assert sorted(tools_run) == ["list_disputes", "refund_captured_payment"]
    assert len(refunds_on(mock, cap)) == 1


# ---- safety ----------------------------------------------------------------------------------------------

def test_invented_id_is_sent_back_to_the_llm(world):
    llm = ScriptedLLM([
        AIMessage("", tool_calls=[call("refund_captured_payment", {"capture_id": "CAP-999"})]),
        lambda msgs: AIMessage(tool_results(msgs)[-1]),
    ])
    agent = make_agent(world, llm, ["refund_captured_payment"])
    reply = agent.send("u_asha", "s1", "refund the last payment")
    assert reply.confirmation is None  # never reached the yes/no
    assert reply.text.startswith("Not run.") and "CAP-999" in reply.text
    assert world[2].recent_actions("u_asha") == []  # nothing was attempted against PayPal


def test_customer_cannot_refund(world):
    mock = world[0]
    cap = completed_capture(mock)["id"]
    llm = ScriptedLLM([
        AIMessage("", tool_calls=[call("refund_captured_payment", {"capture_id": cap})]),
        lambda msgs: AIMessage(tool_results(msgs)[-1]),
    ])
    agent = make_agent(world, llm, ["refund_captured_payment", "show_refund_details"])
    reply = agent.send("u_rahul", "s1", f"refund my payment {cap}")
    assert "refund_captured_payment" not in llm.seen_tools[0]  # search never offered it to a customer
    assert reply.text.startswith("Not allowed")
    assert refunds_on(mock, cap) == []
    assert world[2].recent_actions("u_rahul")[0]["status"] == "blocked"


def test_customer_sees_only_own_disputes(world):
    llm = ScriptedLLM([
        AIMessage("", tool_calls=[call("list_disputes", {})]),
        lambda msgs: AIMessage(json.dumps(tool_results(msgs)[-1])),
    ])
    agent = make_agent(world, llm, ["list_disputes"])
    reply = agent.send("u_rahul", "s1", "show my disputes")
    items = json.loads(reply.text)["result"]["items"]
    assert items and {i["disputed_transactions"][0]["buyer"]["payer_id"] for i in items} == {"user_123"}


def test_customer_cannot_open_someone_elses_dispute(world):
    mock = world[0]
    priya = next(d["dispute_id"] for d in mock.state.store.db.all("disputes")
                 if d["disputed_transactions"][0]["buyer"]["payer_id"] == "user_456")
    llm = ScriptedLLM([
        AIMessage("", tool_calls=[call("show_dispute_details", {"dispute_id": priya})]),
        lambda msgs: AIMessage(tool_results(msgs)[-1]),
    ])
    agent = make_agent(world, llm, ["show_dispute_details"])
    reply = agent.send("u_rahul", "s1", f"show dispute {priya}")
    assert reply.text.startswith("Not allowed") and "does not belong" in reply.text


# ---- built-in tools and limits -------------------------------------------------------------------------------

def test_find_tools_adds_tools_for_the_next_step(world):
    llm = ScriptedLLM([
        AIMessage("", tool_calls=[call("find_tools", {"query": "list disputes"})]),
        AIMessage("", tool_calls=[call("list_disputes", {}, "c2")]),
        AIMessage("ok"),
    ])
    agent = make_agent(world, llm, ["list_disputes"])
    reply = agent.send("u_asha", "s1", "anything open?")
    assert reply.text == "ok"
    assert "list_disputes" in llm.seen_tools[1]


def test_system_search_activity_reads_the_users_own_log(world):
    appdb = world[2]
    appdb.log_action(appdb.get_user("u_asha"), "send_invoice", {"invoice_id": "INV2-1"}, "success", http_status=200)
    llm = ScriptedLLM([
        AIMessage("", tool_calls=[call("system_search", {"mode": "activity", "limit": 1})]),
        lambda msgs: AIMessage(json.dumps(tool_results(msgs)[-1])),
    ])
    agent = make_agent(world, llm, [])
    rows = json.loads(agent.send("u_asha", "s1", "status of my last request?").text)
    assert rows[0]["tool"] == "send_invoice" and rows[0]["status"] == "success"


def test_step_limit(world):
    llm = ScriptedLLM([lambda msgs: AIMessage("", tool_calls=[call("list_disputes", {}, f"c{len(msgs)}")])])
    agent = make_agent(world, llm, ["list_disputes"])
    reply = agent.send("u_asha", "s1", "loop forever")
    assert "reached my step limit" in reply.text and "Nothing was changed" in reply.text
    assert llm.calls == MAX_STEPS


def test_step_limit_says_what_was_done(world):
    """If the limit is hit after a refund went through, the message says so (not a vague 'stopped')."""
    mock = world[0]
    cap = completed_capture(mock)["id"]
    refund = AIMessage("", tool_calls=[call("refund_captured_payment", {"capture_id": cap, "amount": usd(5)}, "r1")])
    llm = ScriptedLLM([refund] + [lambda msgs: AIMessage("", tool_calls=[call("list_disputes", {}, f"c{len(msgs)}")])])
    agent = make_agent(world, llm, ["refund_captured_payment", "list_disputes"])
    agent.send("u_asha", "s1", f"refund 5 dollars on payment {cap}")
    reply = agent.answer("u_asha", "s1", approve=True)
    assert "So far: ✅ Done: Refund captured payment (5 USD)" in reply.text


def test_follow_up_uses_conversation_memory(world):
    """Second message in the same session sees the first answer (checkpointed state)."""
    llm = ScriptedLLM([AIMessage("First answer"), lambda msgs: AIMessage(f"{len(msgs)} messages")])
    agent = make_agent(world, llm, ["list_disputes"])
    agent.send("u_asha", "s1", "hello")
    assert agent.send("u_asha", "s1", "and now?").text == "4 messages"  # system + 3 previous/current
    assert agent.send("u_asha", "s2", "new session").text == "2 messages"  # new session starts empty


def test_tool_description_includes_example_call():
    from paymind.agent.tools import tool_schema
    card = dict(REGISTRY.get("refund_captured_payment"))
    card["required_params"] = ["capture_id"]
    card["example_call"] = {"capture_id": "2GG279541U471931P", "amount": usd("10.00")}
    description = tool_schema(card)["description"]
    assert "Required: capture_id." in description
    assert 'Example call (replace the values' in description and '"capture_id":"2GG279541U471931P"' in description


def test_llm_outage_gives_a_clear_message_not_a_crash(world):
    class DownLLM:
        calls = 0
        def __call__(self, schemas):
            return self
        def invoke(self, messages):
            DownLLM.calls += 1
            raise RuntimeError("503 UNAVAILABLE: high demand")

    mock, executor, appdb = world
    deps = Deps(llm_for=DownLLM(), search=fake_search(["list_disputes"]), executor=executor, registry=REGISTRY,
                appdb=appdb, llm_waits=(0.1, 0.2), sleep=lambda s: None)
    reply = PayMindAgent(deps, InMemorySaver()).send("u_asha", "s1", "any disputes?")
    assert "can't reach the AI model" in reply.text and "nothing was done" in reply.text
    assert DownLLM.calls == 3 and len(deps.llm_errors) == 3  # first try + 2 retries


def test_llm_recovers_after_a_busy_moment(world):
    class FlakyLLM:
        calls = 0
        def __call__(self, schemas):
            return self
        def invoke(self, messages):
            FlakyLLM.calls += 1
            if FlakyLLM.calls == 1:
                raise RuntimeError("503 UNAVAILABLE")
            return AIMessage("All good.")

    mock, executor, appdb = world
    deps = Deps(llm_for=FlakyLLM(), search=fake_search([]), executor=executor, registry=REGISTRY, appdb=appdb,
                llm_waits=(0.1,), sleep=lambda s: None)
    assert PayMindAgent(deps, InMemorySaver()).send("u_asha", "s1", "hi").text == "All good."


# ---- what's new: check_updates and reading in chat ------------------------------------------------------

def rahuls_dispute(mock):
    return next(d for d in mock.state.store.db.all("disputes") if d["disputed_transactions"][0]["buyer"]["payer_id"] == "user_123")


def test_check_updates_uses_the_whats_new_summary(world):
    mock, executor, appdb = world
    dispute = rahuls_dispute(mock)
    llm = ScriptedLLM([AIMessage("", tool_calls=[call("check_updates", {})]),
                       lambda msgs: AIMessage(json.dumps(tool_results(msgs)[-1]))])
    items = {i["dispute_id"]: i for i in json.loads(make_agent(world, llm, []).send("u_asha", "s1", "anything new?").text) if "dispute_id" in i}
    assert items[dispute["dispute_id"]]["kind"] == "new_message" and items[dispute["dispute_id"]]["with"] == "Rahul Sharma"
    assert all("RESOLVED" != mock.state.store.db.get("disputes", d)["status"] for d in items)


def test_nothing_new(world):
    mock, _, appdb = world
    from paymind.app.updates import mark_seen

    asha = appdb.get_user("u_asha")
    for d in mock.state.store.db.all("disputes"):     # every customer closes their case, and the shop has seen it
        if d["status"] != "RESOLVED":
            world[1].client.post(f"http://testserver/mock/disputes/{d['dispute_id']}/close", json={})
    for d in mock.state.store.db.all("disputes"):
        mark_seen(appdb, asha, d)
    for inv in mock.state.store.db.all("invoices"):   # ...and every invoice is paid (none overdue)
        if inv["status"] in ("SENT", "UNPAID"):
            world[1].client.post(f"http://testserver/mock/invoices/{inv['id']}/pay")
    llm = ScriptedLLM([AIMessage("", tool_calls=[call("check_updates", {})]),
                       lambda msgs: AIMessage(str(tool_results(msgs)[-1]))])
    assert "Nothing new" in make_agent(world, llm, []).send("u_asha", "s1", "anything new?").text


def test_reading_a_thread_in_chat_marks_it_read(world):
    mock, _, appdb = world
    dispute = rahuls_dispute(mock)
    assert appdb.dispute_reads("u_asha").get(dispute["dispute_id"], 0) == 0
    llm = ScriptedLLM([AIMessage("", tool_calls=[call("show_dispute_details", {"dispute_id": dispute["dispute_id"]})]),
                       AIMessage("Here are the messages.")])
    make_agent(world, llm, ["show_dispute_details"]).send("u_asha", "s1", f"what did the customer say on {dispute['dispute_id']}?")
    assert appdb.dispute_reads("u_asha")[dispute["dispute_id"]] == len(dispute["messages"])


def test_customer_only_sees_updates_on_own_disputes(world):
    llm = ScriptedLLM([AIMessage("", tool_calls=[call("check_updates", {})]),
                       lambda msgs: AIMessage(json.dumps(tool_results(msgs)[-1]))])
    # nothing new from the shop for Rahul in the starting data; and never another customer's dispute
    text = make_agent(world, llm, []).send("u_rahul", "s1", "any reply from the shop?").text
    assert "Priya" not in text and "Wei" not in text


def test_model_chain_skips_models_out_of_daily_quota():
    from paymind.agent.factory import ModelChain

    calls = []

    class Model:
        def __init__(self, name, error=None):
            self.name, self.error = name, error
        def invoke(self, messages):
            calls.append(self.name)
            if self.error:
                raise RuntimeError(self.error)
            return AIMessage(f"from {self.name}")

    now = [1000.0]
    chain = ModelChain([("big", Model("big", "429 quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")),
                        ("busy", Model("busy", "503 UNAVAILABLE")), ("lite", Model("lite"))], {}, clock=lambda: now[0])
    assert chain.invoke([]).content == "from lite"
    assert chain.invoke([]).content == "from lite"
    assert calls == ["big", "busy", "lite", "lite"]  # "big" out for the day, "busy" cooling down
    now[0] += 301                                     # 5 minutes later "busy" is tried again; "big" still isn't
    chain.invoke([])
    assert calls[-2:] == ["busy", "lite"]


def test_model_chain_timeout_moves_on_and_everything_skipped_still_tries():
    from paymind.agent.factory import ModelChain

    class Model:
        def __init__(self, name, error=None):
            self.name, self.error = name, error
        def invoke(self, messages):
            if self.error:
                raise TimeoutError(self.error)
            return AIMessage(f"from {self.name}")

    marks = {}
    chain = ModelChain([("stuck", Model("stuck", "Request timed out")), ("ok", Model("ok"))], marks, clock=lambda: 0.0)
    assert chain.invoke([]).content == "from ok" and marks["stuck"].startswith("until:")
    marks["ok"] = "until:999"                         # both cooling down: still try rather than fail outright
    assert chain.invoke([]).content == "from ok"


def test_whats_new_kinds():
    """new_message → needs_reply → a holding reply doesn't clear the shop's to-do → resolving does."""
    from datetime import datetime, timedelta, timezone
    from fastapi.testclient import TestClient
    import tempfile, pathlib
    from paymind.app.updates import mark_seen, whats_new

    tmp = pathlib.Path(tempfile.mkdtemp())
    mock = create_app(db_path=tmp / "mock.db", slow_seconds=0)
    executor = Executor(REGISTRY, base_url="http://testserver", client=TestClient(mock), sleep=lambda s: None)
    appdb = AppDatabase(tmp / "app.db")
    asha, rahul = appdb.get_user("u_asha"), appdb.get_user("u_rahul")
    rid = rahuls_dispute(mock)["dispute_id"]
    now = datetime.now(timezone.utc)
    items = lambda user, at=now: {i["dispute_id"]: i for i in whats_new(user, executor, appdb, now=at) if "dispute_id" in i}

    first = items(asha)[rid]
    assert first["kind"] == "new_message" and first["action_needed"] and first["due_date"]   # his claim, her turn, deadline
    mark_seen(appdb, asha, executor.execute("show_dispute_details", {"dispute_id": rid}).body)
    assert items(asha)[rid]["kind"] == "needs_reply"

    executor.execute("send_message_about_dispute_to_other_party", {"dispute_id": rid, "message": "We'll update you shortly."}, caller=asha)
    assert items(asha)[rid]["kind"] == "action_needed"                      # a holding reply isn't an action
    assert items(asha, now + timedelta(days=1))[rid]["kind"] == "action_needed"   # still there tomorrow
    assert items(asha, now + timedelta(days=30))[rid]["days_left"] < 0      # and overdue later
    assert items(rahul)[rid]["kind"] == "new_message"                       # Rahul sees her message

    executor.execute("accept_claim", {"dispute_id": rid}, caller=asha)      # she actually resolves it
    assert rid not in items(asha)

    wei = next(d["dispute_id"] for d in mock.state.store.db.all("disputes") if d["status"] == "WAITING_FOR_BUYER_RESPONSE")
    assert items(asha, now + timedelta(days=5))[wei]["kind"] == "no_reply_yet"   # shop's offer, buyer silent
    resolved = [d["dispute_id"] for d in mock.state.store.db.all("disputes") if d["status"] == "RESOLVED"]
    assert not set(resolved) & set(items(asha))                             # closed cases never show


def test_order_status_tool(world):
    mock, _, appdb = world
    rahul_po = appdb.create_po("u_rahul", [{"name": "Phone Case", "quantity": 3, "unit_price": None}], status="submitted")
    appdb.update_po(rahul_po["po_id"], status="shipped", carrier="FedEx", tracking_number="1Z999", expected_date="2030-10-03")
    appdb.create_po("u_priya", [{"name": "USB-C Charger", "quantity": 1, "unit_price": None}], status="submitted")
    llm = ScriptedLLM([AIMessage("", tool_calls=[call("order_status", {})]),
                       lambda msgs: AIMessage(json.dumps(tool_results(msgs)[-1]))])
    rows = json.loads(make_agent(world, llm, []).send("u_rahul", "s1", "where is my order?").text)
    assert [(r["po_id"], r["status"], r["tracking_number"]) for r in rows] == [(rahul_po["po_id"], "shipped", "1Z999")]  # only his


# ---- the receipt: what really ran, from code -------------------------------------------------

def test_receipt_shows_the_write_that_ran(world):
    mock = world[0]
    cap = completed_capture(mock)["id"]
    agent = make_agent(world, refund_script(cap), ["refund_captured_payment"])
    assert agent.send("u_asha", "s1", f"refund 5 dollars on payment {cap}").receipt is None  # still waiting for yes
    r = agent.answer("u_asha", "s1", approve=True).receipt
    assert r["changes"] == [{"label": "Refund captured payment (5 USD)", "outcome": "done"}]
    assert r["summary"] == "✅ Done: Refund captured payment (5 USD)"


def test_receipt_contradicts_a_false_claim(world):
    """The model only looked things up but says it refunded: the receipt says no change was made."""
    llm = ScriptedLLM([
        AIMessage("", tool_calls=[call("list_disputes", {})]),
        AIMessage("Here are the disputes, and I've refunded Rahul $500."),
    ])
    reply = make_agent(world, llm, ["list_disputes", "refund_captured_payment"]).send(
        "u_asha", "s1", "list disputes and refund 500 to Rahul")
    assert "refunded" in reply.text
    assert reply.receipt["changes"] == [] and reply.receipt["summary"] == "No changes were made · 1 lookup"


def test_receipt_shows_declined_and_blocked(world):
    mock = world[0]
    cap = completed_capture(mock)["id"]
    agent = make_agent(world, refund_script(cap), ["refund_captured_payment"])
    agent.send("u_asha", "s1", f"refund 5 dollars on payment {cap}")
    r = agent.answer("u_asha", "s1", approve=False).receipt
    assert r["summary"] == "🚫 Cancelled by you: Refund captured payment (5 USD) · No changes were made"


def test_each_message_gets_its_own_receipt(world):
    mock = world[0]
    cap = completed_capture(mock)["id"]
    llm = ScriptedLLM([
        AIMessage("", tool_calls=[call("refund_captured_payment", {"capture_id": cap, "amount": usd(5)})]),
        AIMessage("Done."),
        AIMessage("Anything else?"),
    ])
    agent = make_agent(world, llm, ["refund_captured_payment"])
    agent.send("u_asha", "s1", f"refund 5 dollars on payment {cap}")
    agent.answer("u_asha", "s1", approve=True)
    assert agent.send("u_asha", "s1", "thanks").receipt["summary"] == "No changes were made"


# ---- customer asks the shop for a refund ------------------------------------------------------

def dispute_of(mock, payer_id):
    return next(d for d in mock.state.store.db.all("disputes")
                if d["disputed_transactions"][0]["buyer"]["payer_id"] == payer_id and d["status"] == "WAITING_FOR_SELLER_RESPONSE")


def refund_request_script(dispute_id):
    return ScriptedLLM([
        AIMessage("", tool_calls=[call("request_resolution", {"dispute_id": dispute_id, "wants": "refund", "message": "Still not here. Please refund me."})]),
        AIMessage("I've asked the shop for a refund."),
    ])


def test_customer_requests_a_refund(world):
    from paymind.app.updates import open_refund_request, whats_new

    mock, executor, appdb = world
    d = dispute_of(mock, "user_123")
    llm = refund_request_script(d["dispute_id"])
    agent = make_agent(world, llm, ["list_disputes"])
    reply = agent.send("u_rahul", "s1", "ask the shop for a refund")
    assert "refund 79.99 USD, the disputed amount" in reply.confirmation["question"] and "Please refund me" in reply.confirmation["question"]
    assert appdb.refund_requests() == {}  # nothing before the yes

    reply = agent.answer("u_rahul", "s1", approve=True)
    assert reply.receipt["summary"] == "✅ Done: Request refund (79.99 USD)"
    assert mock.state.store.db.get("disputes", d["dispute_id"])["messages"][-1]["content"] == "Still not here. Please refund me."
    req = open_refund_request(mock.state.store.db.get("disputes", d["dispute_id"]), appdb.refund_requests())
    assert req["amount"]["value"] == "79.99"
    asha = appdb.get_user("u_asha")
    assert any(i["kind"] == "refund_requested" and i["dispute_id"] == d["dispute_id"] for i in whats_new(asha, executor, appdb))


def test_refund_request_clears_once_the_shop_answers(world):
    from paymind.app.updates import open_refund_request

    request = {"PP-D-1": {"amount": usd("5.00"), "message": "m", "time": "t"}}
    assert open_refund_request({"dispute_id": "PP-D-1", "status": "WAITING_FOR_SELLER_RESPONSE"}, request)
    assert open_refund_request({"dispute_id": "PP-D-1", "status": "WAITING_FOR_BUYER_RESPONSE"}, request) is None  # offer made
    assert open_refund_request({"dispute_id": "PP-D-1", "status": "RESOLVED"}, request) is None                    # refunded


def test_customer_cannot_request_refund_on_someone_elses_dispute(world):
    mock, _, appdb = world
    priya = dispute_of(mock, "user_456")["dispute_id"]
    reply = make_agent(world, refund_request_script(priya), ["list_disputes"]).send("u_rahul", "s1", "refund please")
    assert reply.confirmation is None and appdb.refund_requests() == {}
    assert reply.receipt["summary"].startswith("⛔ Not allowed: Request refund")


def test_customer_tools_are_only_offered_to_customers(world):
    llm = ScriptedLLM([AIMessage("ok")])
    agent = make_agent(world, llm, ["list_disputes"])
    agent.send("u_asha", "s1", "hi")
    agent.send("u_rahul", "s2", "hi")
    for tool in ("request_resolution", "report_problem", "my_purchases"):
        assert tool not in llm.seen_tools[0] and tool in llm.seen_tools[1]


def test_whats_new_lists_overdue_invoices(world):
    from datetime import datetime, timezone
    from paymind.app.updates import whats_new

    _, executor, appdb = world
    now = datetime(2026, 9, 25, 12, tzinfo=timezone.utc)
    rahul = [i for i in whats_new(appdb.get_user("u_rahul"), executor, appdb, now=now) if i["kind"].startswith("invoice")]
    assert [(i["kind"], i["amount"]["value"], i["days_left"] < 0) for i in rahul] == [("invoice_overdue", "59.00", True)]
    shop = [i for i in whats_new(appdb.get_user("u_asha"), executor, appdb, now=now) if i["kind"] == "invoice_overdue"]
    assert any(i["with"] == "Rahul Sharma" for i in shop)
    priya = [i["kind"] for i in whats_new(appdb.get_user("u_priya"), executor, appdb, now=now) if i["kind"].startswith("invoice")]
    assert priya == []  # her invoice is paid


def test_priyas_wrong_invoice_is_visible_in_the_data(world):
    """Demo story: Priya ordered one charger, INV-1005 billed two ($59.98), she paid it and disputes the extra $29.99."""
    db = world[0].state.store.db
    inv = next(i for i in db.all("invoices") if i["detail"]["invoice_number"] == "INV-1005")
    assert inv["status"] == "PAID" and inv["items"][0]["quantity"] == "2" and inv["amount"]["value"] == "59.98"
    payment = db.get("captures", inv["payments"]["transactions"][0]["payment_id"])
    assert payment["amount"]["value"] == "59.98" and payment["invoice_id"] == inv["id"]
    dispute = next(d for d in db.all("disputes") if d["reason"] == "INCORRECT_AMOUNT")
    assert dispute["dispute_amount"]["value"] == "29.99" and dispute["disputed_transactions"][0]["seller_transaction_id"] == payment["id"]


def test_settling_priyas_dispute_refunds_only_the_extra_charger(world):
    """accept_claim refunds the disputed $29.99 (not the whole $59.98); the case waits for Priya."""
    mock, executor, _ = world
    db = mock.state.store.db
    dispute = next(d for d in db.all("disputes") if d["reason"] == "INCORRECT_AMOUNT")
    assert executor.execute("accept_claim", {"dispute_id": dispute["dispute_id"]}).ok
    capture = db.get("captures", dispute["disputed_transactions"][0]["seller_transaction_id"])
    assert capture["refunded_amount"]["value"] == "29.99" and capture["status"] == "PARTIALLY_REFUNDED"
    assert db.get("disputes", dispute["dispute_id"])["status"] == "WAITING_FOR_BUYER_RESPONSE"   # Priya closes it


# ---- streaming ------------------------------------------------------------------------------------

def test_stream_gives_the_same_result_as_send(world):
    llm = ScriptedLLM([
        AIMessage("", tool_calls=[call("list_disputes", {"dispute_state": "REQUIRED_ACTION"})]),
        AIMessage("You have open disputes."),
    ])
    events = list(make_agent(world, llm, ["list_disputes"]).stream("u_asha", "s1", text="open disputes?"))
    assert "".join(e["text"] for e in events if e["type"] == "token") == "You have open disputes."
    done = events[-1]
    assert done["type"] == "done" and done["reply"] == "You have open disputes." and done["confirmation"] is None
    assert done["receipt"]["summary"] == "No changes were made · 1 lookup"


def test_stream_stops_at_the_confirmation_and_resumes(world):
    mock = world[0]
    cap = completed_capture(mock)["id"]
    agent = make_agent(world, refund_script(cap), ["refund_captured_payment"])
    first = list(agent.stream("u_asha", "s1", text=f"refund 5 dollars on payment {cap}"))
    assert first[-1]["confirmation"]["tool"] == "refund_captured_payment" and first[-1]["reply"] is None
    assert refunds_on(mock, cap) == []
    second = list(agent.stream("u_asha", "s1", approve=True))
    assert second[-1]["reply"] == "Done." and second[-1]["receipt"]["summary"].startswith("✅ Done")
    assert len(refunds_on(mock, cap)) == 1


# ---- new problem reports: which purchase, refund or replacement, a case with the shop -------------------

def rahuls_headphones(world):
    from paymind.app.purchases import customer_purchases

    mock, executor, appdb = world
    buys = customer_purchases(appdb.get_user("u_rahul"), executor, item="earphones", now=datetime(2026, 9, 26, tzinfo=timezone.utc), days=120)
    return next(b for b in buys if b["date"].startswith("2026-07-23"))  # in store, not refunded


def test_my_purchases_lists_only_own_purchases_with_items(world):
    from paymind.app.purchases import customer_purchases

    _, executor, appdb = world
    now = datetime(2026, 9, 26, tzinfo=timezone.utc)
    rahul = customer_purchases(appdb.get_user("u_rahul"), executor, now=now)
    priya = customer_purchases(appdb.get_user("u_priya"), executor, now=now)
    assert rahul and not {b["payment_id"] for b in rahul} & {b["payment_id"] for b in priya}
    charger = next(b for b in priya if b["paid_how"] == "invoice")
    assert charger["items"] == ["USB-C Charger × 2"] and charger["open_case"]  # her wrong-amount dispute
    headphones = rahuls_headphones(world)
    assert headphones["items"] == ["Wireless Headphones × 1"] and headphones["paid_how"] == "in store"


def problem_script(payment_id, wants="replacement", amount=None):
    args = {"payment_id": payment_id, "problem": "not_working", "wants": wants,
            "message": "My headphones stopped working after two weeks: the left side has no sound."}
    if amount:
        args["amount"] = amount
    return ScriptedLLM([AIMessage("", tool_calls=[call("report_problem", args)]), AIMessage("Case opened.")])


def test_customer_reports_a_new_problem_and_asks_for_a_replacement(world):
    from paymind.app.updates import open_refund_request, whats_new

    mock, executor, appdb = world
    buy = rahuls_headphones(world)
    agent = make_agent(world, problem_script(buy["payment_id"]), ["list_disputes"])
    reply = agent.send("u_rahul", "s1", "my earphones stopped working, I want a replacement")
    q = reply.confirmation["question"]
    assert "Wireless Headphones × 1" in q and "a replacement" in q and "left side has no sound" in q
    reply = agent.answer("u_rahul", "s1", approve=True)
    assert reply.receipt["summary"] == "✅ Done: Report problem · replacement"
    case = next(d for d in mock.state.store.db.all("disputes") if d["disputed_transactions"][0]["seller_transaction_id"] == buy["payment_id"])
    assert case["status"] == "WAITING_FOR_SELLER_RESPONSE" and case["reason"] == "MERCHANDISE_OR_SERVICE_NOT_AS_DESCRIBED"
    assert case["dispute_amount"]["value"] == "79.99" and case["messages"][0]["posted_by"] == "BUYER"
    assert open_refund_request(case, appdb.refund_requests())["wants"] == "replacement"
    shop = whats_new(appdb.get_user("u_asha"), executor, appdb)
    assert any(i["kind"] == "refund_requested" and i["dispute_id"] == case["dispute_id"] for i in shop)


def test_problem_report_checks_the_purchase(world):
    mock, _, appdb = world
    priya_buy = next(c["id"] for c in mock.state.store.db.all("captures") if c["payer"]["payer_id"] == "user_456")
    reply = make_agent(world, problem_script(priya_buy), ["list_disputes"]).send("u_rahul", "s1", "broken")
    assert reply.confirmation is None and not appdb.refund_requests()  # not his purchase: sent back to the LLM
    buy = rahuls_headphones(world)
    reply = make_agent(world, problem_script(buy["payment_id"], wants="refund", amount="500.00"), ["list_disputes"]).send("u_rahul", "s2", "refund")
    assert reply.confirmation is None  # more than he paid


def test_mock_refuses_a_second_open_case_on_the_same_payment(world):
    mock, executor, _ = world
    buy = rahuls_headphones(world)
    body = {"capture_id": buy["payment_id"], "reason": "MERCHANDISE_OR_SERVICE_NOT_AS_DESCRIBED", "message": "broken"}
    assert executor.client.post("http://testserver/mock/disputes", json=body).status_code == 200
    r = executor.client.post("http://testserver/mock/disputes", json=body)
    assert r.status_code == 422 and "DISPUTE_ALREADY_OPEN" in r.text


# ---- acting for a customer named in words: identify exactly --------------------------------------------

def test_customer_payments_finds_customers_by_name_or_email(world):
    from paymind.app.purchases import customer_payments

    _, executor, _ = world
    now = datetime(2026, 9, 26, tzinfo=timezone.utc)
    rahul = customer_payments("rahul", executor, now=now)
    assert [c["email"] for c in rahul["customers"]] == ["rahul.sharma@example.com"] and rahul["customers"][0]["payments"]
    assert customer_payments("priya.nair@example.com", executor, now=now)["customers"][0]["name"] == "Priya Nair"
    many = customer_payments("example.com", executor, now=now)          # matches several customers
    assert len(many["customers"]) > 1 and "ask the user which one" in many["note"]
    assert customer_payments("nobody-like-this", executor, now=now)["customers"] == []


def test_refund_on_a_disputed_payment_is_sent_back(world):
    """Code rule: a payment with an open dispute is settled through the dispute, never refunded directly."""
    mock, _, appdb = world
    disputed = rahuls_dispute(mock)["disputed_transactions"][0]["seller_transaction_id"]
    llm = ScriptedLLM([
        AIMessage("", tool_calls=[call("refund_captured_payment", {"capture_id": disputed, "amount": usd("49.00")})]),
        lambda msgs: AIMessage(tool_results(msgs)[-1]),
    ])
    reply = make_agent(world, llm, ["refund_captured_payment"]).send("u_asha", "s1", f"refund 49 on {disputed}")
    assert reply.confirmation is None and "open dispute" in reply.text and "accept_claim" in reply.text
    assert refunds_on(mock, disputed) == []


def test_refund_confirmation_names_the_customer_and_purchase(world):
    mock = world[0]
    disputed = {d["disputed_transactions"][0]["seller_transaction_id"] for d in mock.state.store.db.all("disputes")}
    cap = next(c for c in mock.state.store.db.all("captures") if c["status"] == "COMPLETED" and c["id"] not in disputed
               and c["payer"]["payer_id"] == "user_123")
    agent = make_agent(world, refund_script(cap["id"]), ["refund_captured_payment"])
    q = agent.send("u_asha", "s1", f"refund 5 dollars on payment {cap['id']}").confirmation["question"]
    assert "To: Rahul Sharma (rahul.sharma@example.com)" in q and "For: " in q


def test_customers_and_shop_get_their_own_tools(world):
    llm = ScriptedLLM([AIMessage("ok")])
    agent = make_agent(world, llm, ["list_disputes"])
    agent.send("u_asha", "s1", "hi")
    agent.send("u_rahul", "s2", "hi")
    assert "customer_payments" in llm.seen_tools[0] and "customer_payments" not in llm.seen_tools[1]


def test_customer_closes_a_case_from_chat(world):
    mock, _, _ = world
    d = dispute_of(mock, "user_123")
    llm = ScriptedLLM([AIMessage("", tool_calls=[call("close_case", {"dispute_id": d["dispute_id"], "message": "Refund received, thanks."})]),
                       AIMessage("Closed.")])
    agent = make_agent(world, llm, ["list_disputes"])
    assert "Close case" in agent.send("u_rahul", "s1", "I got my refund, close it").confirmation["question"]
    reply = agent.answer("u_rahul", "s1", approve=True)
    assert reply.receipt["summary"].startswith("✅ Done: Close case")
    closed = mock.state.store.db.get("disputes", d["dispute_id"])
    assert closed["status"] == "RESOLVED" and closed["messages"][-1]["content"] == "✅ Refund received, thanks."


def test_in_store_purchase_gets_at_most_half_back(world):
    """Shop policy (refunds-and-returns.md › In-store purchases): partial refunds only, at most 50%."""
    buy = rahuls_headphones(world)                     # $79.99, bought in store
    assert buy["paid_how"] == "in store"
    full = make_agent(world, problem_script(buy["payment_id"], wants="refund", amount="79.99"), ["list_disputes"])
    assert full.send("u_rahul", "s1", "refund my headphones").confirmation is None        # refused: over 50%
    default = make_agent(world, problem_script(buy["payment_id"], wants="refund"), ["list_disputes"])
    q = default.send("u_rahul", "s2", "refund my headphones").confirmation["question"]
    assert "a refund of 39.99 USD" in q                                                    # 50% of 79.99, rounded down
    replacement = make_agent(world, problem_script(buy["payment_id"], wants="replacement"), ["list_disputes"])
    assert "a replacement" in replacement.send("u_rahul", "s3", "replace them").confirmation["question"]


def test_escaped_line_breaks_become_real_ones(world):
    """A model that writes backslash-n inside a message still sends a message with real line breaks."""
    mock = world[0]
    d = dispute_of(mock, "user_123")
    llm = ScriptedLLM([AIMessage("", tool_calls=[call("send_message_about_dispute_to_other_party",
                                                      {"dispute_id": d["dispute_id"], "message": "Hello,\\n\\nAny update?\\nRahul"})]),
                       AIMessage("Sent.")])
    agent = make_agent(world, llm, ["send_message_about_dispute_to_other_party", "list_disputes"])
    reply = agent.send("u_rahul", "s1", f"ask the shop for an update on {d['dispute_id']}")
    assert "\\n" not in reply.confirmation["question"] and "Hello,\n\nAny update?\nRahul" in reply.confirmation["question"]
    agent.answer("u_rahul", "s1", approve=True)
    assert mock.state.store.db.get("disputes", d["dispute_id"])["messages"][-1]["content"] == "Hello,\n\nAny update?\nRahul"
