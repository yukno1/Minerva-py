from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from minerva.cli.formatter import print_event, safe_echo, safe_secho
from minerva.core.plan_exe import stream_agent_events

app = typer.Typer(help="minerva: a mini Agent.")


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    task: Annotated[
        str | None, typer.Argument(help="Natural-language task for the CodeAgent.")
    ] = None,
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            "-w",
            help="Workspace for generated files. Defaults to .minerva/workspace.",
        ),
    ] = None,
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    configure_console()
    if not task:
        safe_echo(ctx.get_help())
        raise typer.Exit()

    safe_secho(
        "minerva stage 4: MultiAgent + context compression", fg=typer.colors.MAGENTA
    )
    for event in stream_agent_events(task, workspace=workspace):
        print_event(event)
