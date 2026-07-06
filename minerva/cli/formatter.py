from __future__ import annotations

from typing import Any

import typer


def safe_echo(message: Any = "", **kwargs: Any) -> None:
    text = str(message)
    try:
        typer.echo(text, **kwargs)
    except UnicodeEncodeError:
        safe = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        typer.echo(safe, **kwargs)


def safe_secho(message: Any = "", **kwargs: Any) -> None:
    text = str(message)
    try:
        typer.secho(text, **kwargs)
    except UnicodeEncodeError:
        safe = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        typer.secho(safe, **kwargs)


def _shorten(value: Any, limit: int = 260) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def print_event(event: dict[str, Any]) -> None:
    event_type = event.get("type")
    if event_type == "workspace":
        safe_secho(f"workspace: {event['path']}", fg=typer.colors.BLUE)
        return

    if event_type != "agent_event":
        safe_echo(_shorten(event))
        return

    payload = event["event"]
    if not isinstance(payload, dict):
        safe_echo(_shorten(payload))
        return

    for node, update in payload.items():
        safe_secho(f"\n[{node}]", fg=typer.colors.CYAN)
        messages = update.get("messages") if isinstance(update, dict) else None
        if not messages:
            safe_echo(_shorten(update))
            continue
        for message in messages:
            tool_calls = getattr(message, "tool_calls", None)
            name = getattr(message, "name", None)
            content = getattr(message, "content", "")
            if tool_calls:
                for call in tool_calls:
                    safe_secho(
                        f"tool call -> {call.get('name')}", fg=typer.colors.YELLOW
                    )
                    safe_echo(_shorten(call.get("args", {})))
            elif name:
                safe_secho(f"tool result <- {name}", fg=typer.colors.GREEN)
                safe_echo(_shorten(content, 900))
            elif content:
                safe_echo(_shorten(content, 1200))
