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
        self.artifacts = ArtifactStore(self.root)
        self._registry = Registry.load(self.root)
        self._stamp = self._registry_stamp()

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
