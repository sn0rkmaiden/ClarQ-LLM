import argparse
import json
from pathlib import Path

import torch
from dotenv import load_dotenv

from utils.data_loader import data_combination, read_path
from agents.seeker_agent import player
from agents.provider_agent import helpers as general_provider
from agents.multi_info_provider_agent import helpers_m as multi_info_provider
from utils.llm import HookedGEMMA 

def load_state(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"l2l_conv": [], "done": False}

def save_state(path: Path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--task_data_path", type=str, default="data/English")
    ap.add_argument("--i", type=int, required=True, help="type index (same as in l2l.py loop)")
    ap.add_argument("--j", type=int, required=True, help="conv index within that type")
    ap.add_argument("--state_dir", type=str, default="turn_states")
    ap.add_argument("--role", choices=["provider", "seeker"], required=True)

    # provider
    ap.add_argument("--provider_agent_llm", type=str, default="qwen")
    ap.add_argument("--hftoken", type=str, default=None)
    ap.add_argument("--multi_info_provider_agent", action="store_true")

    # seeker (HookedGEMMA)
    ap.add_argument("--player_chat_mode", action="store_true")
    ap.add_argument("--gemma_model_name", type=str, default="gemma-2b-it")
    ap.add_argument("--sae_release", type=str, default="gemma-2b-it-res-jb")
    ap.add_argument("--sae_id", type=str, default="blocks.12.hook_resid_post")
    ap.add_argument("--steering_feature", type=int, default=None)
    ap.add_argument("--steering_strength", type=float, default=1.0)
    ap.add_argument("--steering_max_act", type=float, default=None)
    ap.add_argument("--compute_max_per_turn", action="store_true")

    args = ap.parse_args()

    provider_constructor = multi_info_provider if args.multi_info_provider_agent else general_provider

    # Load one conversation (same as l2l.py)
    all_conv = data_combination(read_path(args.task_data_path))
    conv = all_conv[args.i][args.j]

    state_dir = Path(args.state_dir)
    state_path = state_dir / f"i{args.i}_j{args.j}.json"
    provider_state_path = state_dir / f"i{args.i}_j{args.j}.provider_state.json"

    state = load_state(state_path)
    l2l_conv = state["l2l_conv"]

    torch.cuda.empty_cache()

    if args.role == "provider":
        # Always rebuild helper fresh (new CustomLLM client each run)
        gold_r = conv["all_response"].strip().split("\n")
        h = provider_constructor(
            gold_r,
            conv["background_splitted"],
            conv["gold_structure"],
            conv,
            args.provider_agent_llm,
            api_key=args.hftoken,
        )

        # Restore provider logical state (count + current_levels)
        if provider_state_path.exists():
            ps = json.loads(provider_state_path.read_text(encoding="utf-8"))
            h.count = int(ps.get("count", 0))

            # Restore remaining knowledge state in gold_responses
            levels = ps.get("current_levels", None)
            if levels is not None:
                h.gold.current_levels = set(levels)
                h.gold.current_display = h.gold._get_current_display()

        # Generate provider message
        msg = h.generate_response(l2l_conv)
        l2l_conv.append(msg)

        done = h.is_conv_end(l2l_conv) or len(l2l_conv) > 22

        # Save shared transcript state
        state["l2l_conv"] = l2l_conv
        state["done"] = bool(done)
        save_state(state_path, state)

        # Save provider logical state (JSON-serializable)
        ps_out = {
            "count": h.count,
            "current_levels": sorted(list(h.gold.current_levels)),
        }
        provider_state_path.write_text(
            json.dumps(ps_out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(msg)
        return                  

    if args.role == "seeker":
        if len(l2l_conv) == 0:
            raise RuntimeError("Seeker turn requested but l2l_conv is empty. Run provider first.")

        seeker_llm = HookedGEMMA(
            model_name=args.gemma_model_name,
            sae_release=args.sae_release,
            sae_id=args.sae_id,
            device="cuda",
            steering_feature=args.steering_feature,
            steering_strength=args.steering_strength,
            max_act=args.steering_max_act,
            compute_max_per_turn=args.compute_max_per_turn,
        )

        p = player(conv["background_splitted"], seeker_llm, args.player_chat_mode)

        msg = p.generate_response(l2l_conv)
        l2l_conv.append(msg)

        state["l2l_conv"] = l2l_conv
        save_state(state_path, state)

        print(msg)
        return

if __name__ == "__main__":
    main()
