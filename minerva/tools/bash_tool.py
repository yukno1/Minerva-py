from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from typing import Any

from minerva.core.state import RuntimeState

DEFAULT_TIMEOUT_SECONDS = 10
MAX_OUTPUT_CHARS = 6000

DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\bRemove-Item\b.*\b-Recurse\b.*\b-Force\b",
    r"\bdel\s+/[sq]\b",
    r"\bformat\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r">\s*(?:[A-Za-z]:\\|/)",
]


def run_bash(
    state: RuntimeState,
    command: str,
    timeout_seconds: int | str | float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    if not command.strip():
        return {"ok": False, "error": "command must not be empty"}
    timeout = _coerce_timeout(timeout_seconds)
    if timeout <= 0 or timeout > 60:
        return {"ok": False, "error": "timeout_seconds must be between 1 and 60"}
    normalized_command = _normalize_command(command)

    handled = _handle_tail_command(state, normalized_command)
    if handled is not None:
        return handled

    blocked = _looks_dangerous(normalized_command)
    if blocked:
        return {
            "ok": False,
            "error": f"blocked potentially dangerous command pattern: {blocked}",
        }

    started = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    try:
        completed = subprocess.run(
            normalized_command,
            cwd=state.workspace,
            shell=True,
            # text=True,
            # encoding="utf-8", # cmd子进程是gbk
            # errors="replace",
            capture_output=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "timed_out": True,
            "exit_code": None,
            "stdout": _decode(exc.stdout or "")[:MAX_OUTPUT_CHARS],
            "stderr": _decode(exc.stderr or "")[:MAX_OUTPUT_CHARS],
            "duration_ms": round((time.perf_counter() - started) * 1000),
        }

    stdout = _decode(completed.stdout)[:MAX_OUTPUT_CHARS]
    stderr = _decode(completed.stderr)[:MAX_OUTPUT_CHARS]
    return {
        "ok": completed.returncode == 0,
        "timed_out": False,
        "command": normalized_command,
        "exit_code": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "duration_ms": round((time.perf_counter() - started) * 1000),
    }


def _coerce_timeout(timeout_seconds: int | str | float) -> int:
    try:
        return int(timeout_seconds)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SECONDS


def _normalize_command(command: str) -> str:
    if os.name == "nt":
        normalized = re.sub(
            r"^\s*python3(\.exe)?\b", "python", command, count=1, flags=re.IGNORECASE
        )
        normalized = re.sub(r"\bls\s+-la\b", "dir", normalized)
        normalized = re.sub(r"\bls\b", "dir", normalized)
        normalized = re.sub(r"\bcat\s+([^\s|&<>]+)", r"type \1", normalized)
        return normalized
    return command


def _handle_tail_command(state: RuntimeState, command: str) -> dict[str, Any] | None:
    match = re.fullmatch(r"\s*tail(?:\s+-n)?\s+(\d+)\s+(.+?)\s*", command)
    if not match:
        match = re.fullmatch(r"\s*tail\s+-(\d+)\s+(.+?)\s*", command)
    if not match:
        return None
    count = int(match.group(1))
    raw_path = shlex.split(match.group(2), posix=False)[0]

    from minerva.tools.file_tools import read_text_lossy, resolve_workspace_path

    path = resolve_workspace_path(state, raw_path)
    if not path.exists() or not path.is_file():
        return {"ok": False, "error": f"file does not exist: {raw_path}"}
    lines = read_text_lossy(path).splitlines()
    output = "\n".join(lines[-count:])
    return {
        "ok": True,
        "timed_out": False,
        "command": command,
        "exit_code": 0,
        "stdout": output + ("\n" if output else ""),
        "stderr": "",
        "duration_ms": 0,
    }


def _looks_dangerous(command: str) -> str | None:
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return pattern
    return None


import locale


def _decode(raw: bytes | None) -> str:
    if not raw:
        return ""
    # 优先按 UTF-8 解（Python 子进程 / chcp 65001 场景）
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    # 退回系统默认编码（中文 Windows 上即 GBK/CP936，覆盖 cmd 内建命令）
    enc = locale.getpreferredencoding(False) or "gbk"
    return raw.decode(enc, errors="replace")
