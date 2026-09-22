"""Fetch V* Bench, HR-Bench and Charades-STA into indexed workspaces with question files.

    python -m evals.import_bench vstar        # -> bench/vstar/, 191 questions
    python -m evals.import_bench hrbench_4k   # -> bench/hrbench_4k/, 200 questions
    python -m evals.import_bench hrbench_8k   # -> bench/hrbench_8k/, 200 questions
    python -m evals.import_bench charades_sta --sample 50   # -> bench/charades_sta/

Each workspace gets its images, `questions.jsonl` in the evals.suite format
(kind "choice", answer = option letter), and a registry, so it is ready for
evals.baseline and evals.agent.

HR-Bench ships every question four times with the options rotated
(`cycle_category` 0-3), in TSVs of 1.9 GB and 4.6 GB with the image inlined as
base64. One rotation per question is kept — 200 distinct questions, 100
single-instance (FSP) and 100 cross-instance (FCP). Not rotation 0: there the
answer is always A, and a model that always says A would score 100%. The k-th
question keeps rotation k % 4, so answer letters come out balanced. The TSV is
streamed, never stored, and re-running skips images already on disk.

Charades-STA (Phase 3) is 3,720 sentence-to-moment queries over Charades videos.
The videos sit in four zips totalling ~16 GB; the zips are read over HTTP range
requests and only the sampled videos are extracted. One query per video, drawn
with a fixed seed, as kind "temporal" with answer "start..end" and a tolerance of
TOLERANCE_S seconds per end, written into each question before any run.
Charades is licensed for non-commercial research use only.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import random
import re
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from visru.workspace import video
from visru.workspace.registry import Registry

VSTAR = "craigwu/vstar_bench"
HRBENCH = "DreamMr/HR-Bench"
CHARADES = "lmms-eval/charades_sta"
CHARADES_PARTS = 4
ROWS_API = "https://datasets-server.huggingface.co/rows"
TOLERANCE_S = 1.0
BENCHES = ("vstar", "hrbench_4k", "hrbench_8k", "charades_sta")
OPTION = re.compile(r"^\(([A-Z])\)\s*(.*)$")


def _write_questions(root: Path, rows: list[dict]) -> Path:
    path = root / "questions.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def parse_vstar(record: dict) -> dict:
    """Split V*'s one text field into the question and its lettered options."""
    question, options = [], []
    for line in record["text"].splitlines():
        m = OPTION.match(line.strip())
        if m:
            options.append(m.group(2).strip())
        elif not options:
            question.append(line)
        # anything after the options is the "answer with the letter" instruction
    return {"id": f"vstar_{record['question_id']}", "asset": record["image"],
            "question": "\n".join(question).strip(), "answer": record["label"],
            "kind": "choice", "options": options, "category": record["category"]}


def import_vstar(root: Path) -> list[dict]:
    from huggingface_hub import snapshot_download

    snapshot_download(VSTAR, repo_type="dataset", local_dir=root,
                      allow_patterns=["test_questions.jsonl", "direct_attributes/*",
                                      "relative_position/*"])
    lines = (root / "test_questions.jsonl").read_text().splitlines()
    return [parse_vstar(json.loads(line)) for line in lines if line.strip()]


def parse_hrbench(row: dict, prefix: str) -> dict:
    return {"id": f"{prefix}_{row['index']}", "asset": f"{prefix}_{row['index']}.jpg",
            "question": row["question"].strip(), "answer": row["answer"].strip(),
            "kind": "choice", "options": [row[k].strip() for k in "ABCD"],
            "category": row["category"]}


def import_hrbench(root: Path, which: str) -> list[dict]:
    from huggingface_hub import hf_hub_url

    prefix = "hr4k" if which == "hrbench_4k" else "hr8k"
    url = hf_hub_url(HRBENCH, f"hr_bench_{which.split('_')[1]}.tsv", repo_type="dataset")
    csv.field_size_limit(sys.maxsize)  # one field is a whole base64 8K image
    groups: dict[tuple, dict] = {}  # question -> {"asset": name, "rows": {cycle: row}}
    with urllib.request.urlopen(url) as resp:
        reader = csv.DictReader(io.TextIOWrapper(resp, encoding="utf-8", newline=""),
                                delimiter="\t")
        for n, row in enumerate(reader, start=1):
            image_hash = hashlib.sha1(row["image"].encode()).hexdigest()
            key = (row["question"], frozenset(row[k] for k in "ABCD"), image_hash)
            if key not in groups:
                name = f"{prefix}_{row['index']}.jpg"
                if not (root / name).is_file():
                    (root / name).write_bytes(base64.b64decode(row["image"]))
                groups[key] = {"asset": name, "rows": {}}
            row.pop("image")  # the pixels are on disk; keep only the text
            groups[key]["rows"][int(row["cycle_category"])] = row
            if n % 100 == 0:
                print(f"  {n} rows read, {len(groups)} questions", flush=True)

    out = []
    for k, group in enumerate(groups.values()):
        rows = group["rows"]
        row = rows.get(k % 4) or rows[min(rows)]
        out.append({**parse_hrbench(row, prefix), "asset": group["asset"]})
    return out


def _charades_rows() -> list[dict]:
    """Every test row, via the datasets-server JSON API (no parquet reader needed)."""
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        query = urllib.parse.urlencode({"dataset": CHARADES, "config": "default",
                                        "split": "test", "offset": offset, "length": 100})
        with urllib.request.urlopen(f"{ROWS_API}?{query}") as resp:
            page = json.load(resp)
        total = page["num_rows_total"]
        rows.extend(r["row"] for r in page["rows"])
        offset += len(page["rows"])
        if not page["rows"]:
            break
    return rows


def pick_charades(rows: list[dict], sample: int, seed: int) -> list[dict]:
    """One query per video for `sample` videos, the same ones for the same seed."""
    by_video: dict[str, list[dict]] = {}
    for row in rows:
        by_video.setdefault(row["video"], []).append(row)
    rng = random.Random(seed)
    videos = sorted(by_video)
    chosen = rng.sample(videos, min(sample, len(videos)))
    return [rng.choice(by_video[v]) for v in sorted(chosen)]


def parse_charades(row: dict, duration: float | None = None) -> dict:
    t0, t1 = sorted(float(t) for t in row["timestamp"])
    if duration:  # some annotated ends run past the file; nobody can answer those
        t1 = min(t1, duration)
    caption = row["caption"].strip().rstrip(".")
    return {"id": f"charades_{Path(row['video']).stem}", "asset": row["video"],
            "question": f'In this video, when does this happen: "{caption}"?',
            "answer": f"{t0:.2f}..{t1:.2f}", "kind": "temporal",
            "tolerance": TOLERANCE_S}


def import_charades(root: Path, sample: int, seed: int) -> list[dict]:
    from huggingface_hub import HfFileSystem

    picked = pick_charades(_charades_rows(), sample, seed)
    wanted = {r["video"] for r in picked if not (root / r["video"]).is_file()}
    fs = HfFileSystem()
    for part in range(1, CHARADES_PARTS + 1):
        if not wanted:
            break
        name = f"datasets/{CHARADES}/Charades_v1_480_part_{part}.zip"
        with fs.open(name, "rb", block_size=1 << 20) as remote, \
                zipfile.ZipFile(remote) as z:
            for member in z.namelist():
                base = Path(member).name
                if base in wanted:
                    (root / base).write_bytes(z.read(member))
                    wanted.discard(base)
                    print(f"  {base} from part {part}", flush=True)
    if wanted:
        raise FileNotFoundError(f"not in any zip part: {sorted(wanted)}")
    return [parse_charades(r, video.probe(root / r["video"]).duration) for r in picked]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bench", choices=BENCHES)
    parser.add_argument("--root", default=None, help="defaults to bench/<bench>")
    parser.add_argument("--sample", type=int, default=50,
                        help="charades_sta: videos to fetch, one query each")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    root = Path(args.root or f"bench/{args.bench}").resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.bench == "vstar":
        rows = import_vstar(root)
    elif args.bench == "charades_sta":
        rows = import_charades(root, args.sample, args.seed)
    else:
        rows = import_hrbench(root, args.bench)
    path = _write_questions(root, rows)

    reg = Registry.load(root)
    stats = reg.scan()
    reg.save()
    print(f"{len(rows)} questions -> {path}\n{len(reg.assets)} assets indexed "
          f"({stats.summary()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
