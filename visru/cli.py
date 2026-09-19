from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape

from visru.workspace.registry import Registry

app = typer.Typer(help="visru: visual reasoning and understanding harness")

# Colors match img/visru-logo-acronym.png.
_LETTERS = [
    ("#45DDD0", ["__     __", "\\ \\   / /", " \\ \\ / / ", "  \\ V /  ", "   \\_/   "]),
    ("#808080", [" _ ", "(_)", "| |", "| |", "|_|"]),
    ("#808080", ["     ", " ___ ", "/ __|", "\\__ \\", "|___/"]),
    ("#F9A94A", [" ____  ", "|  _ \\ ", "| |_) |", "|  _ < ", "|_| \\_\\"]),
    ("#D9A6F5", [" _   _ ", "| | | |", "| | | |", "| |_| |", " \\___/ "]),
]
BANNER = "\n".join(
    "".join(f"[{color}]{escape(rows[i])}[/]" for color, rows in _LETTERS)
    for i in range(5)
)


@app.command()
def index(workspace: str, verbose: bool = False) -> None:
    """Build the asset registry for a folder."""
    root = Path(workspace).expanduser().resolve()
    if not root.is_dir():
        raise typer.BadParameter(f"{root} is not a directory")

    reg = Registry.load(root)
    stats = reg.scan()
    reg.save()

    for line in stats.failed:
        typer.secho(f"skipped {line}", fg="yellow", err=True)
    if verbose:
        for asset in reg.list():
            typer.echo(asset.summary())
    typer.echo(f"{len(reg.assets)} assets in {root} ({stats.summary()})")


@app.command()
def run(workspace: str, task: str = typer.Option(..., "--task")) -> None:
    """Run an agent session over a workspace."""
    raise NotImplementedError


@app.command()
def report(session: str) -> None:
    """Render the evidence ledger to a static HTML report."""
    raise NotImplementedError


def main() -> None:
    """Entry point: banner on a terminal, nothing extra when piped."""
    err = Console(stderr=True)
    if err.is_terminal:
        err.print(BANNER, highlight=False)
    app()


if __name__ == "__main__":
    main()
