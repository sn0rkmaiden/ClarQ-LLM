import argparse
import json
import gc
from pathlib import Path

import torch
from utils.data_loader import data_combination, read_path
from agents.seeker_agent import player
from utils.llm import HookedGEMMA


def load_state(p: Path):
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"l2l_conv": [], "done": False}


def save_state(p: Path, s):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_data_path", type=str, default="data/English")
    ap.add_argument("--i", type=int, required=True)
    ap.add_argument("--j", type=int, required=True)

    ap.add_argument("--state_dir", type=str, default="turn_states")
    ap.add_argument("--tag", type=str, default="", help="optional extra suffix")

    ap.add_argument("--features", type=str, required=True,
                    help="comma-separated feature ids, e.g. 771,1023,5001")
    ap.add_argument("--strengths", type=str, default="1.0",
                    help="comma-separated strengths, e.g. 0,0.5,1,2,5")

    # seeker model / SAE
    ap.add_argument("--gemma_model_name", type=str, default="gemma-2-9b-it")
    ap.add_argument("--sae_release", type=str, required=True)
    ap.add_argument("--sae_id", type=str, required=True)

    ap.add_argument("--steering_max_act", type=float, default=None)
    ap.add_argument("--compute_max_per_turn", action="store_true")
    ap.add_argument("--player_chat_mode", action="store_true")

    # generation
    ap.add_argument("--max_new_tokens", type=int, default=128)

    args = ap.parse_args()
    features = parse_int_list(args.features)
    strengths = parse_float_list(args.strengths)

    # Load dataset example
    all_conv = data_combination(read_path(args.task_data_path))
    conv = all_conv[args.i][args.j]

    # Free memory before loading model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # Load model+SAE ONCE
    llm = HookedGEMMA(
        model_name=args.gemma_model_name,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        device="cuda",
        steering_feature=None,  # set per branch
        steering_strength=1.0,  # set per branch
        max_act=args.steering_max_act,
        compute_max_per_turn=args.compute_max_per_turn,
        max_new_tokens=args.max_new_tokens,
    )

    # Player wraps the seeker prompting logic (shared across branches)
    p = player(conv["background_splitted"], llm, args.player_chat_mode)

    base = Path(args.state_dir) / f"i{args.i}_j{args.j}"
    base.mkdir(parents=True, exist_ok=True)

    progressed = 0
    skipped_empty = 0
    skipped_done = 0

    for f in features:
        for s in strengths:
            fname = f"f{f}_s{s}"
            if args.tag:
                fname += f"_{args.tag}"
            state_path = base / f"{fname}.json"

            state = load_state(state_path)
            if state.get("done", False):
                skipped_done += 1
                continue

            l2l_conv = state.get("l2l_conv", [])
            if len(l2l_conv) == 0:
                # Provider must start on local first (Jax greeting)
                skipped_empty += 1
                continue

            # Set steering per branch
            llm.steering_feature = int(f)
            llm.steering_strength = float(s)
            llm.max_act = args.steering_max_act
            llm.compute_max_per_turn = bool(args.compute_max_per_turn)

            # Generate seeker turn
            msg = p.generate_response(l2l_conv)
            l2l_conv.append(msg)

            state["l2l_conv"] = l2l_conv
            save_state(state_path, state)

            progressed += 1
            print(f"[seeker] f={f} s={s}: {msg[:120].replace('\\n',' ')}")

    print(f"\nDone. progressed={progressed}, skipped_empty={skipped_empty}, skipped_done={skipped_done}")


if __name__ == "__main__":
    main()
