from dataclasses import replace
from pathlib import Path

import typer
from rich.console import Console

from visru import backends
from visru.agent import loop as agent_loop
from visru.tools import open_workspace
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
def run(
    workspace: str,
    task: str = typer.Option(..., "--task", help="what to find out"),
    max_steps: int = typer.Option(200, "--max-steps", help="hard ceiling on turns"),
    backend: str | None = typer.Option(None, "--backend",
                                       help="override visru.toml: anthropic | replay"),
    script: str | None = typer.Option(None, "--script",
                                      help="JSON turn script, for --backend replay"),
) -> None:
    """Run an agent session over a workspace.

    Exits non-zero when the session ends without a report: running out of steps is
    a result, but it is not an answer, and a caller scripting this needs to know
    the difference without parsing the output.
    """
    try:
        ctx = open_workspace(workspace)
    except NotADirectoryError as exc:
        raise typer.BadParameter(f"{exc} is not a directory") from None
    if not ctx.registry.assets:
        raise typer.BadParameter(f"no assets in {workspace}; run `visru index` first")

    if backend:
        ctx.cfg = replace(ctx.cfg, model=replace(ctx.cfg.model, backend=backend))
    if script and ctx.cfg.model.backend != "replay":
        raise typer.BadParameter("--script only means something with --backend replay")

    try:
        driver = backends.load(ctx.cfg, **({"path": script} if script else {}))
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from None

    console = Console()
    console.print(f"[dim]{ctx.session} · {ctx.cfg.model.backend} · "
                  f"{len(ctx.registry.assets)} assets[/]")

    session = agent_loop.run(task, ctx, driver, max_steps=max_steps)

    console.print()
    console.print(session.summary())
    if session.usage:
        console.print("[dim]" + "  ".join(f"{k}={v:,}"
                                          for k, v in sorted(session.usage.items())) + "[/]")
    if session.report:
        console.print()
        console.print(session.report)
    console.print(f"\n[dim]trajectory: {session.trajectory}[/]")
    if session.reported:
        console.print(f"[dim]ledger:     "
                      f"{ctx.root / '.visru/ledger' / (ctx.session + '.json')}[/]")
    else:
        console.print(f"[yellow]no report: the session {session.reason}[/]")
        raise typer.Exit(1)


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
