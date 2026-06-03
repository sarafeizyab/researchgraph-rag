from __future__ import annotations

import json
from typing import Any, Iterable


def sse_event(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=True)
    return f"event: {event}\ndata: {payload}\n\n"


def iter_reasoning_trace(trace: Iterable[dict[str, Any]]) -> list[str]:
    events: list[str] = []
    for step in trace:
        events.append(sse_event("reasoning_step", step))
    return events
