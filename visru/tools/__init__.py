"""Tool layer. Plain functions over a workspace, independent of any server.

The MCP server (Phase 0) and the custom harness (Phase 1) both call these, so no
tool may know which one is driving it.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PIL import Image

from visru.config import Config, load_config
from visru.workspace.artifacts import ArtifactStore
from visru.workspace.registry import Registry


def scale_note(cfg: Config, scale: float) -> str:
    """One sentence on whether this render is fine enough to trust.

    Measured, not theoretical: on a 4032 px image the OCR read "LOT 44-871" at 2.0x
    and "LOT 44871" at 0.5x, with 25 phantom lines from out-of-focus blobs.
    """
    inspected = cfg.coverage.inspected_scale
    if scale >= inspected:
        return "at or above inspected_scale"
    return (f"BELOW inspected_scale ({inspected}); zoom into a smaller box before "
            "claiming anything about small objects or text")


@dataclass
class ToolResult:
    """What a tool hands back: text the model reads, images it sees, IDs it can cite."""

    text: str
    images: list[Image.Image] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)


class ToolContext:
    """A workspace, held open across many tool calls."""

    def __init__(self, root: str | Path, cfg: Config, session: str | None = None):
        self.root = Path(root)
        self.cfg = cfg
        self.session = session or f"s_{datetime.now():%Y%m%d_%H%M%S}"
        self.artifacts = ArtifactStore(self.root, self.session)
        self._registry = Registry.load(self.root)
        self._stamp = self._registry_stamp()
        # Set by the agent loop, for `delegate`: the model that runs subagents, and
        # what they spent, so a session's cost includes its children's.
        self.backend = None
        self.child_usage: dict[str, int] = {}
        self.children = 0  # subagent sessions started, so each gets a fresh ID

    def _registry_stamp(self) -> tuple[int, int, int] | None:
        try:
            st = self._registry.state_path.stat()
        except OSError:
            return None  # not indexed yet; the next `visru index` will show up
        # Inode too: `save()` renames a fresh temp file over the old one, so the
        # inode always changes, while mtime alone misses two saves in one clock tick.
        return st.st_mtime_ns, st.st_size, st.st_ino

    @property
    def registry(self) -> Registry:
        """Reload if `visru index` has run since the last call.

        A server started before the first index would otherwise report an empty
        workspace for its whole life, and the user has no reason to suspect the
        registry is a boot-time snapshot.
        """
        if self._registry_stamp() != self._stamp:
            self._registry = Registry.load(self.root)
            self._stamp = self._registry_stamp()
        return self._registry


def resolve_ref(ctx: ToolContext, ref: str) -> tuple[str, list[int] | None]:
    """Resolve an artifact ID to the asset region it was rendered from.

    `view("a_0042")` has to work. The context manager elides an ageing image down
    to a stub naming its artifact ID, and that stub is only honest if the model
    can ask the image back; artifact IDs are not assets, so the registry alone
    cannot resolve them.

    Re-deriving from the parent rather than re-opening the stored PNG keeps one
    coordinate story — what comes back carries a transform into source pixels,
    as the original did — and makes "any view can be remade" true by
    construction rather than by assertion.

    Returns (asset ref, the box it was cropped from). Asset refs pass straight
    through, so every caller can hand us whatever the model said.
    """
    seen: set[str] = set()
    box: list[int] | None = None
    while ref.startswith("a_"):
        if ref in seen:  # a hand-edited sidecar; better than spinning forever
            raise ValueError(f"artifact {ref} is its own ancestor")
        seen.add(ref)
        art = ctx.artifacts.load(ref)
        if art.parent is None:
            raise ValueError(f"{ref} was not rendered from an asset, so there is "
                             "nothing to show; pass an asset ID instead")
        if box is None:  # the outermost artifact is the most specific crop
            box = art.params.get("box")
        ref = art.parent
    return ref, box


def open_workspace(root: str | Path, cfg: Config | None = None,
                   session: str | None = None) -> ToolContext:
    """Open a workspace that `visru index` has already scanned."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if cfg is None:
        local = root / "visru.toml"
        cfg = load_config(local) if local.is_file() else load_config()
    return ToolContext(root, cfg, session)
