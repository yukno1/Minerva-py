from __future__ import annotations

import json
from typing import Any

VALID_TODO_STATUSES = {"pending", "in_progress", "completed", "blocked"}


def write_todos(
    todos: list[str],
    acceptance_criteria: list[str],
    verification_commands: list[str],
) -> dict[str, Any]:
    cleaned_todos = _normalize_items(todos)
    cleaned_criteria = _normalize_items(acceptance_criteria)
    cleaned_commands = _normalize_items(verification_commands)

    return {
        "ok": bool(cleaned_todos and cleaned_criteria and cleaned_commands),
        "todos": cleaned_todos,
        "acceptance_criteria": cleaned_criteria,
        "verification_commands": cleaned_commands,
    }


def update_todo(
    todos: list[dict[str, str]],
    todo_id: str,
    status: str,
    note: str = "",
) -> dict[str, Any]:
    if status not in VALID_TODO_STATUSES:
        return {
            "ok": False,
            "error": f"status must be one of: {', '.join(sorted(VALID_TODO_STATUSES))}",
            "todos": todos,
        }

    updated: list[dict[str, str]] = []
    found = False
    for todo in todos:
        item = dict(todo)
        if item.get("id") == todo_id:
            item["status"] = status
            item["note"] = note
            found = True
        updated.append(item)

    if not found:
        return {"ok": False, "error": f"unknown todo_id: {todo_id}", "todos": todos}
    return {
        "ok": True,
        "todo_id": todo_id,
        "status": status,
        "note": note,
        "todos": updated,
    }


def _normalize_items(items: Any) -> list[str]:
    if isinstance(items, str):
        stripped = items.strip()
        if not stripped:
            return []
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return [
                line.strip("- ").strip()
                for line in stripped.splitlines()
                if line.strip()
            ]
        return _normalize_items(decoded)
    if isinstance(items, dict):
        value = (
            items.get("content")
            or items.get("title")
            or items.get("text")
            or items.get("command")
        )
        return [str(value).strip()] if value else []
    if isinstance(items, list):
        normalized: list[str] = []
        for item in items:
            normalized.extend(_normalize_items(item))
        return [item for item in normalized if item]
    return [str(items).strip()] if items is not None and str(items).strip() else []
