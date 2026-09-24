"""Agent loop tests: a scripted fake LLM drives the real validator, scope checks,
executor, mock PayPal server and audit log. No Gemini quota used."""

import json

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
    assert reply.confirmation["question"] == f"Refund captured payment for 5 USD (capture_id={cap})?"
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
    assert "stopped after several steps" in reply.text
    assert llm.calls == MAX_STEPS


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
    items = {i["dispute_id"]: i for i in json.loads(make_agent(world, llm, []).send("u_asha", "s1", "anything new?").text)}
    assert items[dispute["dispute_id"]]["kind"] == "new_message" and items[dispute["dispute_id"]]["with"] == "Rahul Sharma"
    assert all("RESOLVED" != mock.state.store.db.get("disputes", d)["status"] for d in items)


def test_nothing_new(world):
    mock, _, appdb = world
    for d in mock.state.store.db.all("disputes"):     # the shop resolves every open dispute
        if d["status"] != "RESOLVED":
            world[1].execute("accept_claim", {"dispute_id": d["dispute_id"]}, caller=appdb.get_user("u_asha"))
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

    exhausted = {}
    chain = ModelChain([("big", Model("big", "429 quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")),
                        ("busy", Model("busy", "503 UNAVAILABLE")), ("lite", Model("lite"))], exhausted)
    assert chain.invoke([]).content == "from lite"
    assert chain.invoke([]).content == "from lite"
    assert calls == ["big", "busy", "lite", "busy", "lite"]  # "big" is skipped after its daily quota ran out


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
    items = lambda user, at=now: {i["dispute_id"]: i for i in whats_new(user, executor, appdb, now=at)}

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
