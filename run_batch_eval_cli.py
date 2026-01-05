#!/usr/bin/env python3
"""CLI wrapper around ClarQ-LLM's batch_eval.evaluate_files.

This script lets you run batch_eval on one or more result JSON files and
save the aggregated metrics to disk (CSV/JSON).

Examples
--------
# Evaluate two files and save CSV
python run_batch_eval_cli.py \
  --repo_root /path/to/ClarQ-LLM-main \
  --inputs results/a.json results/b.json \
  --evaluation_set 5 \
  --llm qwen --api_key $QWEN_KEY \
  --out metrics.csv

# Evaluate all JSONs under a directory
python run_batch_eval_cli.py \
  --repo_root /path/to/ClarQ-LLM-main \
  --inputs /path/to/results_dir \
  --glob '**/*.json' \
  --evaluation_set 5 \
  --out metrics.csv
"""

import argparse
import json
import sys
from pathlib import Path


def collect_json_files(inputs: list[str], glob_pat: str) -> list[str]:
    files: list[Path] = []
    for x in inputs:
        p = Path(x)
        if p.is_dir():
            files.extend(sorted(p.glob(glob_pat)))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(f"Input not found: {x}")

    # De-duplicate while preserving order
    seen = set()
    out: list[str] = []
    for f in files:
        s = str(f.resolve())
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo_root",
        type=str,
        required=True,
        help="Path to ClarQ-LLM repo root (folder that contains batch_eval.py).",
    )
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more JSON result files and/or directories containing JSON files.",
    )
    ap.add_argument(
        "--glob",
        type=str,
        default="**/*.json",
        help="Glob pattern used when an input is a directory (default: **/*.json).",
    )
    ap.add_argument(
        "--evaluation_set",
        type=str,
        default="3",
        help='Evaluation set spec passed to batch_eval (e.g., "5", "0-5", "0,2,4").',
    )
    ap.add_argument("--llm", type=str, default="qwen", help="Judge LLM name.")
    ap.add_argument("--api_key", type=str, default="", help="API key for the judge LLM (if needed).")
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output file path. Extension can be .csv or .json (or use --format).",
    )
    ap.add_argument(
        "--format",
        type=str,
        choices=["csv", "json"],
        default=None,
        help="Output format override (otherwise inferred from --out extension).",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    if not repo_root.exists():
        print(f"ERROR: repo_root does not exist: {repo_root}", file=sys.stderr)
        return 2

    # Make repo importable
    sys.path.insert(0, str(repo_root))

    try:
        from batch_eval import evaluate_files  # type: ignore
    except Exception as e:
        print("ERROR: failed to import batch_eval.evaluate_files", file=sys.stderr)
        print(f"  repo_root={repo_root}", file=sys.stderr)
        print(f"  exception={e}", file=sys.stderr)
        return 2

    json_paths = collect_json_files(args.inputs, args.glob)
    if not json_paths:
        print("ERROR: no JSON files found from inputs.", file=sys.stderr)
        return 2

    print(f"Found {len(json_paths)} JSON files")

    df = evaluate_files(
        json_paths=json_paths,
        llm_name=args.llm,
        api_key=args.api_key,
        evaluation_set=args.evaluation_set,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = args.format
    if fmt is None:
        ext = out_path.suffix.lower()
        fmt = "csv" if ext != ".json" else "json"

    if fmt == "csv":
        df.to_csv(out_path, index=False)
    else:
        # Records-style JSON
        out_path.write_text(json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
