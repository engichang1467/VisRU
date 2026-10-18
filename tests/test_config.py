import pytest

from visru.config import load_config, with_model
from visru.geometry import fit_render_size


def test_defaults_need_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg.path is None
    assert cfg.model.name == "claude-sonnet-5"
    # limits must unpack straight into fit_render_size
    assert fit_render_size(4032, 3024, **cfg.model.limits)[0] <= cfg.model.max_edge


def test_overrides_merge(tmp_path, monkeypatch):
    (tmp_path / "visru.toml").write_text('[model]\nname = "gpt-x"\nmax_edge = 1568\n')
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg.model.name == "gpt-x"
    assert cfg.model.max_edge == 1568
    assert cfg.model.patch == 28  # untouched default
    assert cfg.context.max_live_images == 16


def test_typos_raise(tmp_path):
    p = tmp_path / "visru.toml"
    p.write_text("[model]\nmax_edg = 1568\n")
    with pytest.raises(ValueError, match="max_edg"):
        load_config(p)

    p.write_text("[modell]\nmax_edge = 1568\n")
    with pytest.raises(ValueError, match="modell"):
        load_config(p)


def test_explicit_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.toml")


def test_env_picks_backend_without_toml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VISRU_BACKEND", "openai")
    monkeypatch.setenv("VISRU_MODEL", "gpt-x")
    cfg = load_config()
    assert cfg.model.backend == "openai_compatible"
    assert cfg.model.base_url == "https://api.openai.com/v1"
    assert cfg.model.name == "gpt-x"


def test_explicit_base_url_beats_preset(tmp_path, monkeypatch):
    (tmp_path / "visru.toml").write_text('[model]\nname = "from-toml"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VISRU_MODEL", "from-env")
    cfg = with_model(load_config(), backend="openai", base_url="http://localhost:8000/v1")
    assert cfg.model.backend == "openai_compatible"
    assert cfg.model.base_url == "http://localhost:8000/v1"
    assert cfg.model.name == "from-env"  # env beats toml
