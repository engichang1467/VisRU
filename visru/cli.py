from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from visru import backends
from visru.agent import loop as agent_loop
from visru.config import with_model
from visru.evidence import report as evidence_report
from visru.evidence.ledger import Ledger
from visru.tools import open_workspace
from visru.workspace.registry import Registry

app = typer.Typer(help="visru: visual reasoning and understanding harness")

# Model options shared by every command that runs the loop.
BackendOpt = Annotated[str | None, typer.Option(
    "--backend",
    help="override visru.toml and $VISRU_BACKEND: anthropic | openai | openai_compatible | replay")]
ModelOpt = Annotated[str | None, typer.Option("--model", help="override visru.toml and $VISRU_MODEL")]
BaseUrlOpt = Annotated[str | None, typer.Option(
    "--base-url", help="openai_compatible endpoint; overrides $VISRU_BASE_URL")]
ScriptOpt = Annotated[str | None, typer.Option("--script",
                                               help="JSON turn script, for --backend replay")]

# What `visru init` lets the survey touch: looking, not measuring or checking.
SURVEY_TOOLS = ["list_assets", "view", "sample_frames", "search", "ocr", "submit_report"]
INIT_PROMPT = Path(agent_loop.__file__).parent / "prompts" / "init.md"

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
def index(
    workspace: str,
    verbose: bool = False,
    embed: bool = typer.Option(False, "--embed",
                               help="also build the SigLIP index `search` reads "
                                    "(needs the [tools] extra)"),
    embed_fps: float = typer.Option(1.0, "--embed-fps",
                                    help="video frames embedded per second"),
) -> None:
    """Build the asset registry for a folder, and optionally its embedding index."""
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

    if embed:
        from visru.workspace import index as embedding_index

        built, added = embedding_index.build(
            root, reg, fps=embed_fps,
            progress=lambda a: typer.echo(f"  embedded {a.name}") if verbose else None)
        typer.echo(f"{len(built)} vectors from {len(built.assets())} assets "
                   f"({added} newly embedded, {built.model})")


@app.command()
def init(
    workspace: str,
    no_model: bool = typer.Option(False, "--no-model",
                                  help="registry facts only; no survey, no API spend"),
    max_steps: int = typer.Option(15, "--max-steps", help="ceiling on survey turns"),
    backend: BackendOpt = None,
    model: ModelOpt = None,
    base_url: BaseUrlOpt = None,
    script: ScriptOpt = None,
) -> None:
    """Index a folder, survey it, and draft its VISRU.md.

    An existing VISRU.md is never overwritten; the draft goes to VISRU.md.new.
    Exits non-zero when the survey ends without a report, after still writing
    the registry facts.
    """
    from visru.workspace import survey

    root = Path(workspace).expanduser().resolve()
    if not root.is_dir():
        raise typer.BadParameter(f"{root} is not a directory")

    reg = Registry.load(root)
    stats = reg.scan()
    reg.save()
    for line in stats.failed:
        typer.secho(f"skipped {line}", fg="yellow", err=True)

    existing = root / "VISRU.md"
    claims, session_id, surveyed = [], None, True
    if not no_model and reg.assets:
        ctx = open_workspace(root)
        driver = _driver(ctx, backend, model, base_url, script)
        task = "Survey this workspace for its VISRU.md.\n\n" + survey.facts(reg)
        if existing.is_file():
            task += ("\n\nIts current VISRU.md is below. Check its observations against "
                     "the pixels rather than repeating them.\n\n" + existing.read_text())
        typer.echo(f"surveying {len(reg.assets)} assets "
                   f"({ctx.session} · {ctx.cfg.model.backend})", err=True)
        result = agent_loop.run(
            task, ctx, driver, max_steps=max_steps, tools=SURVEY_TOOLS,
            system=agent_loop.system_prompt() + "\n\n" + INIT_PROMPT.read_text())
        surveyed = result.reported
        if surveyed:
            session_id = ctx.session
            claims = Ledger.load(root, session_id, ctx.artifacts).claims
        else:
            typer.secho(f"survey {result.reason} without a report; writing registry "
                        f"facts only (trajectory: {result.trajectory})",
                        fg="yellow", err=True)

    out = existing
    if out.exists():
        out = root / "VISRU.md.new"
        typer.secho(f"VISRU.md exists; wrote {out.name} to diff against it", fg="yellow",
                    err=True)
    out.write_text(survey.draft(reg, claims, session_id))
    typer.echo(f"{out} ({len(reg.assets)} assets, {len(claims)} observations)")
    if not surveyed:
        raise typer.Exit(1)


def _driver(ctx, backend: str | None, model: str | None, base_url: str | None,
            script: str | None):
    """Apply the model flags to the workspace config and load the backend."""
    ctx.cfg = with_model(ctx.cfg, backend=backend, name=model, base_url=base_url)
    if script and ctx.cfg.model.backend != "replay":
        raise typer.BadParameter("--script only means something with --backend replay")
    try:
        return backends.load(ctx.cfg, **({"path": script} if script else {}))
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from None


@app.command()
def run(
    workspace: str,
    task: str = typer.Option(..., "--task", help="what to find out"),
    max_steps: int = typer.Option(30, "--max-steps", help="hard ceiling on turns"),
    backend: BackendOpt = None,
    model: ModelOpt = None,
    base_url: BaseUrlOpt = None,
    script: ScriptOpt = None,
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

    driver = _driver(ctx, backend, model, base_url, script)

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
        console.print(f"[dim]report:     {evidence_report.render(ctx.root, ctx.session)}[/]")
    else:
        console.print(f"[yellow]no report: the session {session.reason}[/]")
        raise typer.Exit(1)


@app.command()
def report(
    workspace: str,
    session: str | None = typer.Argument(None, help="defaults to the latest ledger"),
) -> None:
    """Render a session's evidence ledger to a static HTML report."""
    root = Path(workspace).expanduser().resolve()
    if session is None:
        found = evidence_report.sessions(root)
        if not found:
            raise typer.BadParameter(f"no ledgers in {root}; run `visru run` first")
        session = found[-1]
    try:
        out = evidence_report.render(root, session)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from None
    typer.echo(out)


def main() -> None:
    """Entry point: banner on a terminal, nothing extra when piped."""
    err = Console(stderr=True)
    if err.is_terminal:
        err.print(BANNER, highlight=False)
    app()


if __name__ == "__main__":
    main()
