from pathlib import Path

import typer

from visru.workspace.registry import Registry

app = typer.Typer(help="visru: visual reasoning and understanding harness")


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


if __name__ == "__main__":
    app()
