"""Shared HTTP helper: one session, polite rate limiting, bounded retries.

NCBI caps unauthenticated clients at 3 requests/sec and will hand back HTTP 429
if you exceed it, so every fetcher in this package goes through `get()`.
"""

from __future__ import annotations

import time
import requests

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "medical-literature-rag/0.1 (academic project)"})

_last_call: dict[str, float] = {}


def get(url: str, params: dict | None = None, *, min_interval: float = 0.34,
        timeout: int = 30, retries: int = 3) -> requests.Response:
    """GET with a per-host minimum interval between calls and exponential backoff."""
    host = url.split("/")[2]
    elapsed = time.monotonic() - _last_call.get(host, 0.0)
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, params=params, timeout=timeout)
            _last_call[host] = time.monotonic()
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"GET failed after {retries} attempts: {url}") from last_err
