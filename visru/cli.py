from pathlib import Path

import typer
from rich.console import Console

from visru.workspace.registry import Registry

app = typer.Typer(help="visru: visual reasoning and understanding harness")

# Generated from img/visru-logo-acronym.png
BANNER = "\n".join([
    '[#45DDD0]▀████▀▀[/]  [#45DDD0]▀▀██▀▀[/]         [#F9A94A]▀████▀▀███▄▄[/] [#D9A6F5]▀▀████▀[/]   [#D9A6F5]▀███▀[/]',
    '  [#45DDD0]███▄[/]    [#45DDD0]▄█[/]              [#F9A94A]███[/]    [#F9A94A]███[/]   [#D9A6F5]███[/]       [#D9A6F5]█[/]  ',
    '   [#45DDD0]███[/]    [#45DDD0]█▀[/]    [#808080]██[/]        [#F9A94A]███[/]   [#F9A94A]▄██▀[/]   [#D9A6F5]███[/]       [#D9A6F5]█[/]  ',
    '   [#45DDD0]▀███[/]  [#45DDD0]██[/]    [#808080]▄▄[/]   [#808080]▄▄▄[/]   [#F9A94A]████████[/]     [#D9A6F5]███[/]       [#D9A6F5]█[/]  ',
    '    [#45DDD0]███[/] [#45DDD0]▄█[/]     [#808080]██[/]  [#808080]██[/] [#808080]▀[/]   [#F9A94A]███[/]   [#F9A94A]███▄[/]   [#D9A6F5]███[/]       [#D9A6F5]█[/]  ',
    '     [#45DDD0]████▀[/]     [#808080]█▀[/]   [#808080]▀█▄[/]   [#F9A94A]███[/]    [#F9A94A]███[/]   [#D9A6F5]▀███[/]     [#D9A6F5]██[/]  ',
    '      [#45DDD0]███[/]      [#808080]██▀▀█▄█▀[/] [#F9A94A]███████[/]  [#F9A94A]▀████[/]  [#D9A6F5]▀▀██████▀[/]   ',
])


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
