import pytest
from pydantic import ValidationError

from paymind.ingest.enricher import (
    EnrichmentBatch,
    ToolEnrichment,
    batches,
    build_user_prompt,
    enrich_batch,
    merge,
)


def raw_tool(name, category="invoices", method="POST", action_type="write"):
    return {
        "name": name,
        "title": name.replace("_", " ").title(),
        "service": "paypal",
        "category": category,
        "subcategory": None,
        "method": method,
        "path": f"/v2/{category}/{{id}}",
        "description": name,
        "needs_description": True,
        "parameters": {"type": "object", "properties": {"id": {"type": "string", "x-in": "path"}}, "required": ["id"]},
        "example_request_body": None,
        "action_type": action_type,
        "supports_idempotency": True,
    }


def enrichment(name, action_type="write", roles=("accountant",)):
    return ToolEnrichment(
        name=name,
        description=f"Does the {name} thing for a customer.",
        example_questions=["first question", "second question", "second question", "  third question  "],
        action_type=action_type,
        allowed_roles=list(roles),
    )


def test_merge_keeps_facts_and_adds_language():
    raw = raw_tool("send_invoice")
    card = merge(raw, enrichment("send_invoice", roles=("accountant", "customer", "accountant")))

    for fact in ("name", "method", "path", "parameters", "category", "supports_idempotency"):
        assert card[fact] == raw[fact]
    assert card["description"] == "Does the send_invoice thing for a customer."
    assert card["example_questions"] == ["first question", "second question", "third question"]
    assert card["allowed_roles"] == ["customer", "accountant"]
    assert card["requires_confirmation"] is True
    assert card["enrichment_status"] == "ok"
    assert card["eval_only"] is False
    assert "needs_description" not in card


def test_merge_confirmation_follows_reviewed_action_type():
    card = merge(raw_tool("generate_qr_code"), enrichment("generate_qr_code", action_type="read"))
    assert card["action_type"] == "read"
    assert card["requires_confirmation"] is False


def test_merge_failed_falls_back_to_raw_and_safe_role():
    raw = raw_tool("delete_invoice")
    card = merge(raw, None)
    assert card["description"] == raw["description"]
    assert card["allowed_roles"] == ["accountant"]
    assert card["requires_confirmation"] is True
    assert card["enrichment_status"] == "failed"


def test_enrich_batch_handles_missing_and_renamed_tools():
    tools = [raw_tool("send_invoice"), raw_tool("cancel_sent_invoice")]

    def fake_llm(system, user):
        assert "send_invoice" in user and "cancel_sent_invoice" in user
        return EnrichmentBatch(tools=[enrichment("send_invoice"), enrichment("cancel_invoice_renamed")])

    cards = enrich_batch(fake_llm, "invoices", tools)
    assert [c["enrichment_status"] for c in cards] == ["ok", "failed"]
    assert [c["name"] for c in cards] == ["send_invoice", "cancel_sent_invoice"]


def test_batches_group_by_category_and_split_large_ones():
    tools = [raw_tool(f"inv_{i}") for i in range(14)] + [raw_tool("list_disputes", category="disputes")]
    result = batches(tools, size=12)
    assert [(cat, len(group)) for cat, group in result] == [("invoices", 12), ("invoices", 2), ("disputes", 1)]


def test_prompt_shows_what_llm_needs_only():
    prompt = build_user_prompt("invoices", [raw_tool("send_invoice")])
    assert '"name": "send_invoice"' in prompt
    assert '"parameters": [\n      "id"\n    ]' in prompt
    assert "example_request_body" not in prompt


@pytest.mark.parametrize("bad", [
    {"example_questions": ["only one"]},
    {"allowed_roles": []},
    {"allowed_roles": ["admin"]},
    {"action_type": "maybe"},
])
def test_invalid_llm_output_is_rejected(bad):
    data = enrichment("x").model_dump() | bad
    with pytest.raises(ValidationError):
        ToolEnrichment(**data)


def test_retry_then_fallback_on_transient_errors():
    from paymind.ingest.enricher import with_retry_and_fallback

    calls, waits = [], []
    good = EnrichmentBatch(tools=[])

    def busy(system, user):
        calls.append("busy")
        raise RuntimeError("503 UNAVAILABLE: high demand")

    def flaky_then_ok(system, user):
        calls.append("backup")
        if calls.count("backup") < 2:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return good

    enrich = with_retry_and_fallback([("main", busy), ("backup", flaky_then_ok)], waits=(1, 2), sleep=waits.append)
    assert enrich("s", "u") is good
    assert calls == ["busy", "busy", "busy", "backup", "backup"]
    assert waits == [1, 2, 1]


def test_non_transient_errors_are_not_retried():
    from paymind.ingest.enricher import with_retry_and_fallback

    def broken(system, user):
        raise ValueError("400 INVALID_ARGUMENT")

    enrich = with_retry_and_fallback([("main", broken)], waits=(1,), sleep=lambda s: None)
    with pytest.raises(ValueError):
        enrich("s", "u")


def test_daily_quota_skips_model_for_rest_of_run():
    from paymind.ingest.enricher import with_retry_and_fallback

    calls, waits = [], []
    good = EnrichmentBatch(tools=[])

    def out_of_quota(system, user):
        calls.append("main")
        raise RuntimeError("429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    def backup(system, user):
        calls.append("backup")
        return good

    enrich = with_retry_and_fallback([("main", out_of_quota), ("backup", backup)], waits=(1, 2), sleep=waits.append)
    assert enrich("s", "u") is good
    assert enrich("s", "u") is good
    assert calls == ["main", "backup", "backup"]  # main tried once, never again
    assert waits == []


def test_call_example_optional_body_is_allowed():
    """Some writes take an optional body (e.g. capture the full amount with an empty body)."""
    import json as _json
    from paymind.agent.executor import ToolRegistry
    from paymind.ingest.call_examples import CallExample, check_example

    card = ToolRegistry().get("capture_authorized_payment")
    answer = CallExample(name=card["name"], required_params=[], example_call_json=_json.dumps({"authorization_id": "0VF52814937998046"}))
    required, example, problems = check_example(card, answer)
    assert problems == [] and required == ["authorization_id"]  # path params are always required


def test_call_example_is_tried_on_a_sandbox_mock():
    """An example that matches the schema but that PayPal would reject is caught by the dry run."""
    import json as _json
    from paymind.agent.executor import ToolRegistry
    from paymind.ingest.call_examples import CallExample, check_example, sandbox_dry_run

    card, dry = ToolRegistry().get("create_draft_invoice"), sandbox_dry_run()
    weak = CallExample(name=card["name"], required_params=[],
                       example_call_json=_json.dumps({"detail": {"currency_code": "USD"}}))
    assert any("mock PayPal rejected" in p for p in check_example(card, weak, dry)[2])
    good = CallExample(name=card["name"], required_params=["primary_recipients", "items"], example_call_json=_json.dumps({
        "primary_recipients": [{"billing_info": {"email_address": "john@x.com"}}],
        "items": [{"name": "Consulting", "quantity": "1", "unit_amount": {"currency_code": "USD", "value": "50.00"}}]}))
    assert check_example(card, good, dry)[2] == []
