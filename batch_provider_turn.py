import argparse
import json
import copy
from pathlib import Path

import torch
from dotenv import load_dotenv

from utils.data_loader import data_combination, read_path
from agents.provider_agent import helpers as general_provider
from agents.multi_info_provider_agent import helpers_m as multi_info_provider


def load_state(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def save_state(p: Path, s):
    p.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")


def get_gold_obj(h):
    # Support both provider helper variants in your repo
    if hasattr(h, "gold"):
        return h.gold
    if hasattr(h, "gold_responses"):
        return h.gold_responses
    raise AttributeError("Provider helper has no gold object (expected h.gold or h.gold_responses).")


def main():
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--task_data_path", type=str, default="data/English")
    ap.add_argument("--i", type=int, required=True)
    ap.add_argument("--j", type=int, required=True)

    ap.add_argument("--state_dir", type=str, default="turn_states")

    ap.add_argument("--provider_agent_llm", type=str, default="qwen")
    ap.add_argument("--hftoken", type=str, default=None)
    ap.add_argument("--multi_info_provider_agent", action="store_true")

    ap.add_argument("--max_turns_cap", type=int, default=22,
                    help="hard cap on conversation length for safety")

    args = ap.parse_args()

    provider_constructor = multi_info_provider if args.multi_info_provider_agent else general_provider

    # Load dataset example
    all_conv = data_combination(read_path(args.task_data_path))
    conv = all_conv[args.i][args.j]

    base = Path(args.state_dir) / f"i{args.i}_j{args.j}"
    if not base.exists():
        raise FileNotFoundError(f"Branch directory not found: {base}")

    # Loop through all branches
    branch_files = sorted(base.glob("f*_s*.json"))
    if len(branch_files) == 0:
        raise FileNotFoundError(f"No branch state files found in {base} (expected f*_s*.json)")

    progressed = 0
    skipped_done = 0

    for state_path in branch_files:
        state = load_state(state_path)
        if state.get("done", False):
            skipped_done += 1
            continue

        l2l_conv = state.get("l2l_conv", [])

        conv_local = copy.deepcopy(conv)

        gold_r = conv_local["all_response"].strip().split("\n")
        h = provider_constructor(
            gold_r,
            conv_local["background_splitted"],
            conv_local["gold_structure"],
            conv_local,
            args.provider_agent_llm,
            api_key=args.hftoken,
        )

        gold = get_gold_obj(h)

        # Restore provider logical state (local-only file)
        provider_state_path = state_path.with_suffix(".provider_state.json")
        if provider_state_path.exists():
            ps = json.loads(provider_state_path.read_text(encoding="utf-8"))
            if hasattr(h, "count") and ps.get("count", None) is not None:
                h.count = int(ps["count"])
            levels = ps.get("current_levels", None)
            if levels is not None:
                gold.current_levels = set(levels)
                gold.current_display = gold._get_current_display()

        # Provider generates next message (turn 0 greeting if l2l_conv is empty)
        msg = h.generate_response(l2l_conv)
        l2l_conv.append(msg)

        done = h.is_conv_end(l2l_conv) or len(l2l_conv) > args.max_turns_cap

        state["l2l_conv"] = l2l_conv
        state["done"] = bool(done)
        save_state(state_path, state)

        # Save provider logical state (JSON-serializable; local-only)
        ps_out = {
            "count": getattr(h, "count", None),
            "current_levels": sorted(list(gold.current_levels)),
        }
        provider_state_path.write_text(json.dumps(ps_out, ensure_ascii=False, indent=2), encoding="utf-8")

        progressed += 1
        pretty = msg[:120].replace("\n", " ")
        print(f"[provider] {state_path.name}: {pretty}")

        torch.cuda.empty_cache()

    print(f"\nDone. progressed={progressed}, skipped_done={skipped_done}")


if __name__ == "__main__":
    main()
