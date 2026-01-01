import argparse
import json
from pathlib import Path


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state_dir", default="turn_states")
    ap.add_argument("--i", type=int, required=True)
    ap.add_argument("--j", type=int, required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--strengths", required=True)
    args = ap.parse_args()

    feats = parse_int_list(args.features)
    strengths = parse_float_list(args.strengths)

    base = Path(args.state_dir) / f"i{args.i}_j{args.j}"
    base.mkdir(parents=True, exist_ok=True)

    created = 0
    for f in feats:
        for s in strengths:
            p = base / f"f{f}_s{s}.json"
            if not p.exists():
                p.write_text(json.dumps({"l2l_conv": [], "done": False}, ensure_ascii=False, indent=2),
                             encoding="utf-8")
                created += 1
    print(f"Created {created} branch files in {base}")


if __name__ == "__main__":
    main()
