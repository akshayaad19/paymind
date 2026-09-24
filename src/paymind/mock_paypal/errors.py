"""Errors in PayPal's JSON shape, so the agent sees what real PayPal would send."""

from __future__ import annotations

import uuid

from fastapi import Request
from fastapi.responses import JSONResponse

NAMES = {400: "INVALID_REQUEST", 404: "RESOURCE_NOT_FOUND", 422: "UNPROCESSABLE_ENTITY", 500: "INTERNAL_SERVER_ERROR", 503: "SERVICE_UNAVAILABLE"}


class PayPalError(Exception):
    def __init__(self, status: int, issue: str, description: str, field: str | None = None):
        self.status, self.issue, self.description, self.field = status, issue, description, field


def error_body(status: int, issue: str, description: str, field: str | None = None) -> dict:
    detail = {"issue": issue, "description": description}
    if field:
        detail["field"] = field
    return {
        "name": NAMES.get(status, "ERROR"),
        "message": description,
        "debug_id": uuid.uuid4().hex[:13],
        "details": [detail],
    }


async def paypal_error_handler(request: Request, exc: PayPalError) -> JSONResponse:
    return JSONResponse(error_body(exc.status, exc.issue, exc.description, exc.field), status_code=exc.status)


def not_found(kind: str, resource_id: str) -> PayPalError:
    return PayPalError(404, "INVALID_RESOURCE_ID", f"No {kind} found with ID {resource_id}.")
