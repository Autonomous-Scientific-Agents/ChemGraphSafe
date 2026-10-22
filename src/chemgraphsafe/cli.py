"""CLI workflows. No command sends paid requests unless its config selects a live model."""

import json
from pathlib import Path
from typing import Annotated

import httpx
import typer

from . import data, runner
from .config import load_config
from .reporting import report as make_report
from .storage import exclusive

app = typer.Typer(no_args_is_help=True, help="Chemistry request-triage research workflows.")
data_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data")


def execute(action, config: Path):
    """Load config and run ``action`` under exclusive dataset (and run) locks.

    Translates ``ValueError``, ``OSError``, and ``httpx.HTTPError`` into a clean
    error message on stderr with exit code 1. Returns exit code 2 when a run
    completes but with errors (incomplete inference).
    """
    try:
        c = load_config(config)
        # Dataset lock also prevents concurrent annotate/freeze/evaluate races on the same dir.
        with exclusive(c.dataset / ".workflow.lock"):
            if action in (runner.evaluate, runner.smoke):
                with exclusive(c.run / ".workflow.lock"):
                    result = action(c)
            else:
                result = action(c)
        typer.echo(json.dumps(result, indent=2))
        if result.get("complete") is False:
            raise typer.Exit(2)
    except (ValueError, OSError, httpx.HTTPError) as exc:
        # HTTP exception strings can contain URLs; endpoints never contain secrets, and
        # response bodies / model-generated text are never printed here.
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@data_app.command("prepare")
def prepare(config: Annotated[Path, typer.Option(exists=True)]):
    execute(data.prepare, config)


@data_app.command("freeze")
def freeze(config: Annotated[Path, typer.Option(exists=True)]):
    execute(data.freeze, config)


@app.command()
def curate(config: Annotated[Path, typer.Option(exists=True)]):
    execute(runner.curate, config)


@app.command()
def evaluate(config: Annotated[Path, typer.Option(exists=True)]):
    execute(runner.evaluate, config)


@app.command()
def smoke(config: Annotated[Path, typer.Option(exists=True)]):
    execute(runner.smoke, config)


@app.command()
def report(run: Annotated[Path, typer.Option(exists=True)]):
    try:
        with exclusive(run / ".workflow.lock"):
            typer.echo(json.dumps(make_report(run), indent=2))
    except (ValueError, OSError, KeyError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
