from __future__ import annotations

from langchain_core.tools import StructuredTool

from minerva.core.state import RuntimeState
from .file_tools import read_file, write_file, edit_file
from .bash_tool import run_bash
from .grep_tool import grep

__all__ = ["get_tools"]


def get_tools(state: RuntimeState) -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            name="FileReadTool",
            func=lambda file_path, offset=0, limit=2000: read_file(
                state, file_path, offset, limit
            ),
            description="Read a UTF-8 text file inside the workspace. Supports offset and limit.",
        ),
        StructuredTool.from_function(
            name="FileWriteTool",
            func=lambda file_path, content: write_file(state, file_path, content),
            description="Create a new file or rewrite an existing file inside the workspace.",
        ),
        StructuredTool.from_function(
            name="FileEditTool",
            func=lambda file_path, old_text, new_text: edit_file(
                state, file_path, old_text, new_text
            ),
            description="Edit an existing workspace file by replacing one unique old_text snippet.",
        ),
        StructuredTool.from_function(
            name="GrepTool",
            func=lambda pattern,
            path=".",
            glob=None,
            head_limit=50,
            ignore_case=False: grep(
                state, pattern, path, glob, head_limit, ignore_case
            ),
            description="Search workspace text files by regex pattern and return matching lines.",
        ),
        StructuredTool.from_function(
            name="BashTool",
            func=lambda command, timeout_seconds=10: run_bash(
                state, command, timeout_seconds
            ),
            description="Run a safe development shell command inside the workspace with timeout and output capture.",
        ),
    ]
