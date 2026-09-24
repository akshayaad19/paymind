"""Put the mock PayPal data back to its starting state (a developer command, not an app feature).

    python -m paymind.mock_paypal.reset

If the mock server is running, it's asked to reset itself (POST /mock/reset), so
it reloads cleanly. If it isn't running, data/mock/initial.db is copied over the
working database directly.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[3]
INITIAL_DB = ROOT / "data/mock/initial.db"
WORKING_DB = ROOT / "data/mock/paypal_mock.db"


def reset(base_url: str, working_db: Path = WORKING_DB, initial_db: Path = INITIAL_DB) -> str:
    try:
        httpx.post(f"{base_url.rstrip('/')}/mock/reset", timeout=5).raise_for_status()
        return f"Reset through the running mock server at {base_url}."
    except httpx.HTTPError:
        working_db.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(initial_db, working_db)
        return f"Mock server not reachable; copied {initial_db.name} over {working_db.name}."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.getenv("PAYPAL_BASE_URL", "http://localhost:8000"))
    print(reset(ap.parse_args().url))


if __name__ == "__main__":
    main()
