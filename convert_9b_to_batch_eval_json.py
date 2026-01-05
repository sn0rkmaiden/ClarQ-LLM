#!/usr/bin/env python3
"""Convert 9B ClarQ-LLM steering dumps into the JSON schema expected by batch_eval.py.

Input 9B schema (per run):
    { "l2l_conv": [...], "done": true/false }

batch_eval.py expects a JSON file shaped like:
    all_conv: List[one_type]
    one_type: List[conv]
    conv has keys like: all_response, all_response_exaplain, l2l, ...

We rebuild conv metadata from the official ClarQ-LLM dataset (utils.data_loader),
and insert l2l_conv into conv['l2l'][0] for the matching (i, j).

Notes:
- batch_eval.py computes denom = 10 * len(evaluation_set). If you only evaluated a single
  dialogue (e.g., i5_j0), the metrics will be scaled down unless you modify batch_eval.py.
"""

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Tuple


DIALOG_RE = re.compile(r"i(\d+)_j(\d+)")
RUN_RE = re.compile(r"f(\d+)_s([0-9.]+)\.json$")


def iter_run_files(results_root: Path) -> Iterable[Path]:
    for p in results_root.rglob("*.json"):
        if p.name.endswith(".provider_state.json"):
            continue
        if RUN_RE.search(p.name):
            yield p


def parse_ij(path: Path) -> Tuple[int, int]:
    for part in path.parts:
        m = DIALOG_RE.fullmatch(part)
        if m:
            return int(m.group(1)), int(m.group(2))
    raise ValueError(f"Could not find i*_j* in path: {path}")


def parse_feat_alpha(filename: str) -> Tuple[int, float]:
    m = RUN_RE.search(filename)
    if not m:
        raise ValueError(f"Could not parse feature/alpha from filename: {filename}")
    return int(m.group(1)), float(m.group(2))


def load_base_all_conv(repo_root: Path, data_dir: Path):
    sys.path.insert(0, str(repo_root))
    from utils.data_loader import read_path, data_combination  # type: ignore

    all_data = read_path(str(data_dir))
    return data_combination(all_data)


def ensure_l2l(conv: dict):
    l2l = conv.get("l2l")
    if not isinstance(l2l, list) or len(l2l) < 2:
        conv["l2l"] = [[], []]
    if not isinstance(conv["l2l"][0], list):
        conv["l2l"][0] = []
    if not isinstance(conv["l2l"][1], list):
        conv["l2l"][1] = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, required=True,
                    help="Path to ClarQ-LLM repo root (contains utils/data_loader.py)")
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Dataset directory, e.g. <repo_root>/data/English or .../data/Chinese")
    ap.add_argument("--results_root", type=str, required=True,
                    help="Root of 9B result dumps (contains .../i*_j*/f*_s*.json)")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Where to write converted JSON files")
    ap.add_argument("--exclude_dialogues", type=str, default="",
                    help="Comma-separated dialogue IDs to skip (e.g., i6_j0)")
    ap.add_argument("--only_type", type=int, default=None,
                    help="If set, convert only runs whose i==ONLY_TYPE.")
    ap.add_argument("--trim", action="store_true",
                    help="If set, output only the touched type(s) (smaller files). Default writes full 31x10 dataset.")
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    data_dir = Path(args.data_dir)
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude = {s.strip() for s in args.exclude_dialogues.split(",") if s.strip()}

    base_all_conv = load_base_all_conv(repo_root, data_dir)
    n_types = len(base_all_conv)

    runs = list(iter_run_files(results_root))
    if not runs:
        raise SystemExit(f"No run files found under {results_root}")

    for run_path in runs:
        i, j = parse_ij(run_path)
        if args.only_type is not None and i != args.only_type:
            continue
        dial_id = f"i{i}_j{j}"
        if dial_id in exclude:
            continue

        feat, alpha = parse_feat_alpha(run_path.name)

        run = json.loads(run_path.read_text(encoding="utf-8"))
        l2l_conv = run.get("l2l_conv", [])
        if not isinstance(l2l_conv, list):
            l2l_conv = []

        if args.trim:
            out_all_conv = [[] for _ in range(n_types)]
            out_all_conv[i] = copy.deepcopy(base_all_conv[i])
        else:
            out_all_conv = copy.deepcopy(base_all_conv)

        try:
            conv = out_all_conv[i][j]
        except Exception as e:
            raise SystemExit(f"Index (i={i}, j={j}) not valid for dataset loaded from {data_dir}: {e}")

        ensure_l2l(conv)
        conv["l2l"][0] = l2l_conv

        # provenance (ignored by batch_eval)
        conv["_converted_from"] = str(run_path)
        conv["_feature"] = feat
        conv["_alpha"] = alpha
        conv["_dialogue_id"] = dial_id

        out_name = f"batch_eval_i{i}_j{j}_f{feat}_s{alpha}.json"
        out_path = out_dir / out_name
        out_path.write_text(json.dumps(out_all_conv, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote converted files to: {out_dir}")


if __name__ == "__main__":
    main()
