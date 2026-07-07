from __future__ import annotations

from pathlib import Path

from mokioclaw.core.state import RuntimeState
from mokioclaw.tools.bash_tool import bash_tool_description, run_bash
from mokioclaw.tools.file_tools import edit_file, read_file, write_file
from mokioclaw.tools.grep_tool import grep
from mokioclaw.tools.todo_tool import update_todo, write_todos
from mokioclaw.tools.web_search_tool import web_search


def make_state(tmp_path: Path) -> RuntimeState:
    return RuntimeState(workspace=tmp_path)


def test_read_file_records_snapshot(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    (tmp_path / "demo.py").write_text("one\ntwo\nthree\n", encoding="utf-8")

    result = read_file(state, "demo.py", limit=2)

    assert result["ok"] is True
    assert result["total_lines"] == 3
    assert "1: one" in result["content"]
    assert state.snapshot_for(tmp_path / "demo.py") is not None


def test_read_file_accepts_gbk_text(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    (tmp_path / "demo_output.txt").write_bytes("错误：需要 curses 库\n".encode("gbk"))

    result = read_file(state, "demo_output.txt")

    assert result["ok"] is True
    assert "curses" in result["content"]


def test_workspace_prefix_is_collapsed(tmp_path: Path) -> None:
    state = make_state(tmp_path)

    result = write_file(state, "workspace/snake_game.py", "print('ok')\n")

    assert result["ok"] is True
    assert (tmp_path / "snake_game.py").exists()
    assert not (tmp_path / "workspace" / "snake_game.py").exists()


def test_write_file_creates_new_file(tmp_path: Path) -> None:
    state = make_state(tmp_path)

    result = write_file(state, "hello.py", "print('hello')\n")

    assert result["ok"] is True
    assert result["type"] == "create"
    assert (tmp_path / "hello.py").read_text(encoding="utf-8") == "print('hello')\n"


def test_write_file_returns_error_for_path_outside_workspace(tmp_path: Path) -> None:
    state = make_state(tmp_path)

    result = write_file(state, "../outside.py", "print('nope')\n")

    assert result["ok"] is False
    assert "inside workspace" in result["error"]
    assert not (tmp_path.parent / "outside.py").exists()


def test_write_file_requires_read_before_overwrite(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    (tmp_path / "hello.py").write_text("print('old')\n", encoding="utf-8")

    result = write_file(state, "hello.py", "print('new')\n")

    assert result["ok"] is False
    assert "not been read" in result["error"]


def test_edit_file_replaces_unique_text(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    (tmp_path / "hello.py").write_text("name = 'old'\nprint(name)\n", encoding="utf-8")
    read_file(state, "hello.py")

    result = edit_file(state, "hello.py", "old", "new")

    assert result["ok"] is True
    assert "new" in (tmp_path / "hello.py").read_text(encoding="utf-8")


def test_edit_file_rejects_multiple_matches(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    (tmp_path / "hello.py").write_text("x = 1\nx = 2\n", encoding="utf-8")
    read_file(state, "hello.py")

    result = edit_file(state, "hello.py", "x", "y")

    assert result["ok"] is False
    assert "matched 2 times" in result["error"]


def test_grep_finds_matches(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    (tmp_path / "a.py").write_text("class Snake:\n    pass\n", encoding="utf-8")

    result = grep(state, "Snake")

    assert result["ok"] is True
    assert result["matches"][0]["path"] == "a.py"
    assert result["matches"][0]["line"] == 1


def test_bash_runs_command_inside_workspace(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    (tmp_path / "hello.py").write_text("print('hello from smoke')\n", encoding="utf-8")

    result = run_bash(state, "python hello.py", timeout_seconds=5)

    assert result["ok"] is True
    assert "hello from smoke" in result["stdout"]


def test_bash_accepts_string_timeout(tmp_path: Path) -> None:
    state = make_state(tmp_path)

    result = run_bash(state, "python --version", timeout_seconds="5")

    assert result["ok"] is True


def test_bash_sets_utf8_for_python_subprocess(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    (tmp_path / "emoji.py").write_text("print('🎮')\n", encoding="utf-8")

    result = run_bash(state, "python emoji.py", timeout_seconds=5)

    assert result["ok"] is True
    assert "🎮" in result["stdout"]


def test_bash_supports_tail_file_on_windows_style_usage(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    (tmp_path / "demo_output.txt").write_text("1\n2\n3\n", encoding="utf-8")

    result = run_bash(state, "tail -2 demo_output.txt", timeout_seconds=5)

    assert result["ok"] is True
    assert result["stdout"] == "2\n3\n"


def test_bash_normalizes_workspace_cd_and_pwd(tmp_path: Path) -> None:
    state = make_state(tmp_path)

    result = run_bash(state, "cd /workspace && pwd", timeout_seconds=5)

    assert result["ok"] is True
    assert str(tmp_path) in result["stdout"]
    assert result["command"] == "cd"


def test_bash_blocks_dangerous_command(tmp_path: Path) -> None:
    state = make_state(tmp_path)

    result = run_bash(state, "rm -rf .", timeout_seconds=5)

    assert result["ok"] is False
    assert "blocked" in result["error"]


def test_bash_tool_description_mentions_windows_cmd(monkeypatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Windows")

    description = bash_tool_description()

    assert "cmd.exe" in description
    assert "Do not use POSIX-only tools" in description


def test_bash_tool_description_mentions_posix_for_macos(monkeypatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Darwin")

    description = bash_tool_description()

    assert "macOS" in description
    assert "POSIX shell" in description


def test_bash_tool_description_mentions_posix_for_linux(monkeypatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Linux")

    description = bash_tool_description()

    assert "Linux/Unix" in description
    assert "POSIX shell" in description


def test_todo_write_tool_records_plan_parts() -> None:
    result = write_todos(
        ["write tests", "implement"],
        ["tests pass"],
        ["python -m pytest -q"],
    )

    assert result["ok"] is True
    assert result["todos"] == ["write tests", "implement"]
    assert result["acceptance_criteria"] == ["tests pass"]
    assert result["verification_commands"] == ["python -m pytest -q"]


def test_todo_write_tool_normalizes_json_strings() -> None:
    result = write_todos(
        '[{"title": "write tests"}, {"title": "implement"}]',
        "- tests pass\n- demo runs",
        '["python -m pytest -q"]',
    )

    assert result["todos"] == ["write tests", "implement"]
    assert result["acceptance_criteria"] == ["tests pass", "demo runs"]
    assert result["verification_commands"] == ["python -m pytest -q"]


def test_todo_update_tool_updates_existing_todo() -> None:
    todos = [
        {"id": "todo-1", "content": "write tests", "status": "pending", "note": ""}
    ]

    result = update_todo(todos, "todo-1", "completed", "tests written")

    assert result["ok"] is True
    assert result["todos"][0]["status"] == "completed"
    assert result["todos"][0]["note"] == "tests written"


def test_todo_update_tool_rejects_unknown_todo() -> None:
    todos = [
        {"id": "todo-1", "content": "write tests", "status": "pending", "note": ""}
    ]

    result = update_todo(todos, "todo-2", "completed")

    assert result["ok"] is False
    assert result["todos"][0]["status"] == "pending"


def test_web_search_tool_requires_tavily_key(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "")

    result = web_search("Amiya Arknights")

    assert result["ok"] is False
    assert "TAVILY_API_KEY" in result["error"]


def test_web_search_tool_parses_tavily_results(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key

        def search(self, **kwargs):
            return {
                "answer": "Amiya is from Arknights.",
                "results": [
                    {
                        "title": "Amiya",
                        "url": "https://example.com/amiya",
                        "content": "Amiya profile",
                        "score": 0.9,
                    }
                ],
            }

    monkeypatch.setattr("tavily.TavilyClient", FakeClient)

    result = web_search("Amiya Arknights")

    assert result["ok"] is True
    assert result["answer"] == "Amiya is from Arknights."
    assert result["results"][0]["url"] == "https://example.com/amiya"
