"""Config loading. Defaults are the whole config; visru.toml only overrides."""

import os
import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    backend: str = "anthropic"  # or "openai_compatible"
    name: str = "claude-sonnet-5"
    # openai_compatible only. The HF router by default; point it at a local vLLM
    # to run the same model on your own GPU. Pin the provider in `name`
    # ("Qwen/Qwen3-VL-30B-A3B-Instruct:novita"): tool support differs between
    # providers serving the same weights.
    base_url: str = "https://router.huggingface.co/v1"
    # 2000 not 2576: once a request holds more than 20 images every image must be
    # <=2000 px per side, and any real session crosses that.
    max_edge: int = 2000
    max_visual_tokens: int = 4784
    patch: int = 28
    max_upscale: float = 2.0
    # Room to think and still emit a few tool calls. Lowballing this truncates a
    # turn mid-thought and costs a whole retry.
    max_tokens: int = 16_000
    # openai_compatible only. Left unset, vLLM samples at its own defaults and an
    # open model in a long image context degenerates into repeating one sentence
    # until it hits max_tokens. These are Qwen's own recommended values; the
    # penalty is what actually stops the loop.
    temperature: float = 0.7
    top_p: float = 0.8
    presence_penalty: float = 1.0

    @property
    def limits(self) -> dict[str, Any]:
        """Unpacks straight into geometry.fit_render_size."""
        return {
            "max_edge": self.max_edge,
            "max_visual_tokens": self.max_visual_tokens,
            "patch": self.patch,
            "max_upscale": self.max_upscale,
        }


@dataclass(frozen=True)
class ContextConfig:
    max_live_images: int = 16  # stays under the 20-image threshold
    elide_batch: int = 8  # elide in batches to keep the cached prefix stable
    compact_at_tokens: int = 150_000


@dataclass(frozen=True)
class CoverageConfig:
    inspected_scale: float = 1.0  # rendered px per source px that counts as "inspected"


@dataclass(frozen=True)
class Config:
    model: ModelConfig = ModelConfig()
    context: ContextConfig = ContextConfig()
    coverage: CoverageConfig = CoverageConfig()
    path: Path | None = None  # where it was loaded from, None for defaults


def _apply(section: Any, overrides: dict[str, Any], name: str) -> Any:
    known = {f.name for f in fields(section)}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"unknown keys in [{name}]: {', '.join(sorted(unknown))}")
    return replace(section, **overrides)


_ENV = {"VISRU_BACKEND": "backend", "VISRU_MODEL": "name", "VISRU_BASE_URL": "base_url"}
# Shorthand backend names: each expands to a real backend plus the fields it implies.
PRESETS = {"openai": {"backend": "openai_compatible", "base_url": "https://api.openai.com/v1"}}


def with_model(cfg: Config, **overrides: str | None) -> Config:
    """Override [model] fields, expanding a preset backend name. None means unset."""
    overrides = {k: v for k, v in overrides.items() if v}
    if preset := PRESETS.get(overrides.get("backend")):
        # An explicit base_url still beats the preset's; the backend name does not.
        overrides = {**preset, **overrides, "backend": preset["backend"]}
    return replace(cfg, model=replace(cfg.model, **overrides)) if overrides else cfg


def load_config(path: str | Path | None = None) -> Config:
    """Load visru.toml (explicit path, else ./visru.toml, else built-in defaults),
    then apply $VISRU_BACKEND, $VISRU_MODEL and $VISRU_BASE_URL on top."""
    env = {field: os.environ.get(var) for var, field in _ENV.items()}
    return with_model(_load_toml(path), **env)


def _load_toml(path: str | Path | None) -> Config:
    p = Path(path) if path else Path("visru.toml")
    if not p.is_file():
        if path is not None:
            raise FileNotFoundError(p)
        return Config()

    raw = tomllib.loads(p.read_text())
    cfg = Config(path=p)
    unknown = set(raw) - {"model", "context", "coverage"}
    if unknown:
        raise ValueError(f"unknown sections in {p}: {', '.join(sorted(unknown))}")
    for name in ("model", "context", "coverage"):
        if name in raw:
            cfg = replace(cfg, **{name: _apply(getattr(cfg, name), raw[name], name)})
    return cfg
