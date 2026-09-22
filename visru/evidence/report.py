"""The ledger as a static HTML page: each claim beside the pixels that back it.

One file, no server, no JS. Images are linked relative to the report, not
inlined, so the page stays small and points at the same PNGs the ledger cites.
"""

import json
from pathlib import Path

from jinja2 import Environment

from visru.evidence.ledger import LEDGER_DIR, STATUSES
from visru.workspace.artifacts import ARTIFACT_DIR

REPORT_DIR = ".visru/reports"

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>visru {{ session }}</title>
<style>
:root { --bg: #fff; --fg: #1a1a1a; --muted: #666; --line: #ddd;
  --verified: #1a7f37; --supported: #0969da; --contested: #bf8700; --hypothesis: #6e7781; }
@media (prefers-color-scheme: dark) { :root { --bg: #0d1117; --fg: #e6edf3; --muted: #8b949e;
  --line: #30363d; --verified: #3fb950; --supported: #58a6ff; --contested: #d29922;
  --hypothesis: #8b949e; } }
body { background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, sans-serif;
  max-width: 1100px; margin: 0 auto; padding: 16px; }
h1 { font-size: 1.4em; } h2 { font-size: 1.1em; border-bottom: 1px solid var(--line); }
.claim { border: 1px solid var(--line); border-radius: 6px; padding: 12px; margin: 12px 0; }
.status { font-weight: 600; text-transform: uppercase; font-size: .8em; }
{% for s in statuses %}.{{ s }} { color: var(--{{ s }}); }{% endfor %}
.muted { color: var(--muted); font-size: .9em; }
.thumbs { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
.thumbs figure { margin: 0; } .thumbs img { max-width: 260px; max-height: 200px;
  border: 1px solid var(--line); display: block; }
figcaption { font: 12px monospace; color: var(--muted); }
table { border-collapse: collapse; width: 100%; } td, th { text-align: left;
  padding: 4px 8px; border-bottom: 1px solid var(--line); font-size: .9em; }
code { font-size: .9em; }
</style></head><body>
<h1>Evidence report <code>{{ session }}</code></h1>
<p class="muted">{{ claims|length }} claims · created {{ created }}</p>
{% for status in statuses %}{% set group = claims|selectattr("status", "eq", status)|list %}
{% if group %}
<h2 class="{{ status }}">{{ "Hypotheses: no evidence" if status == "hypothesis"
  else status|capitalize }} ({{ group|length }})</h2>
{% for c in group %}<div class="claim">
<div><span class="status {{ status }}">{{ status }}</span> <code>{{ c.claim_id }}</code>
  {{ c.text }}</div>
{% for k in c.checks %}<div class="muted">check {{ k.check or "" }}: {{ k.kind }}
  {{ k.methods|join(" vs ") }} → <b>{{ k.result }}</b>{% if k.iou is defined %}
  (IoU {{ k.iou }}){% endif %}{% if not k.check %} · written by the model, not
  computed{% endif %}</div>{% endfor %}
<div class="thumbs">{% for a in c.evidence + (c.checks|map(attribute="check")|select|list) %}
<figure><a href="{{ href(a) }}"><img src="{{ href(a) }}" alt="{{ a }}" loading="lazy"></a>
<figcaption>{{ a }} {{ kinds.get(a, "") }}</figcaption></figure>{% endfor %}</div>
</div>{% endfor %}{% endif %}{% endfor %}

<h2>Coverage</h2>
{% if coverage.assets %}<p class="muted">At inspected_scale = {{ coverage.inspected_scale }}
  rendered px per source px.</p>
<table><tr><th>asset</th><th>inspected</th><th>seen at any scale</th></tr>
{% for aid, c in coverage.assets.items() %}<tr><td><code>{{ aid }}</code></td>
{% if c.kind == "image" %}<td>{{ "%.0f"|format(c.inspected * 100) }}%</td>
<td>{{ "%.0f"|format(c.seen * 100) }}%</td>
{% else %}<td>{{ c.frames_inspected }} / {{ c.frames }} frames</td>
<td>{{ c.frames_seen }} frames</td>{% endif %}</tr>{% endfor %}</table>
{% else %}<p class="muted">No coverage recorded.</p>{% endif %}
{% if coverage.untouched %}<p>Never viewed: {{ coverage.untouched|length }} assets
<span class="muted">{{ coverage.untouched[:50]|join(", ") }}{% if coverage.untouched|length > 50 %}
…{% endif %}</span></p>{% endif %}
</body></html>
"""


def sessions(root: str | Path) -> list[str]:
    """Sessions with a saved ledger, oldest first."""
    ledgers = (Path(root) / LEDGER_DIR).glob("*.json")
    return [p.stem for p in sorted(ledgers, key=lambda p: p.stat().st_mtime)]


def render(root: str | Path, session: str) -> Path:
    """Write `.visru/reports/<session>.html` from the saved ledger. Returns its path."""
    root = Path(root).resolve()
    ledger_path = root / LEDGER_DIR / f"{session}.json"
    if not ledger_path.is_file():
        raise FileNotFoundError(f"no ledger for session {session!r} at {ledger_path}")
    raw = json.loads(ledger_path.read_text())

    kinds = {}
    for claim in raw["claims"]:
        for c in claim["checks"]:
            c.setdefault("check", None)
            c.setdefault("kind", "unspecified")
            c.setdefault("methods", [])
            c.setdefault("result", "")
        cited = claim["evidence"] + [c["check"] for c in claim["checks"] if c["check"]]
        for a in cited:
            sidecar = root / ARTIFACT_DIR / f"{a}.json"
            if sidecar.is_file():
                kinds[a] = json.loads(sidecar.read_text())["kind"]

    html = Environment(autoescape=True).from_string(TEMPLATE).render(
        session=raw["session"], created=raw.get("created", ""), claims=raw["claims"],
        coverage=raw.get("coverage") or {}, statuses=STATUSES, kinds=kinds,
        href=lambda a: f"../artifacts/{a}.png")
    out = root / REPORT_DIR / f"{session}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out
