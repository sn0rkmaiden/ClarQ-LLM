from agents.seeker_agent import player
from agents.provider_agent import helpers as general_provider
from agents.multi_info_provider_agent import helpers_m as multi_info_provider
from ALL_KEYS import *
from utils.data_loader import *
from utils.llm import ChatGPT, QianFan, LLAMA, AWSBedrockLLAMA, CustomLLM, HookedGEMMA, HuggingFaceLLM
import argparse
import torch
from dotenv import load_dotenv


def evaluate_player(task_data_path, output_path, player_llm, player_chat_mode, provider_constructor, provider_llm, hftoken, evaluation_set):
    all_conv = data_combination(read_path(task_data_path)) # len(all_conv) = 31
    # evaluation_set = [i for i in range(26)]
    # evaluation_set = [0, 1, 2]
    # evaluate_results = []
    result_idx = 0

    for i, one_type in enumerate(all_conv):
        if i not in evaluation_set:
            continue
        # evaluate_results.append([])
        for j, conv in enumerate(one_type):

            print("{0}.{1}".format(i+1,j))
            # evaluate_results[result_idx].append([])

            torch.cuda.empty_cache()

            gold_r = conv['all_response'].strip().split('\n')            
            h = provider_constructor(gold_r, conv['background_splitted'], conv['gold_structure'], conv, provider_llm, api_key=hftoken)            
            p = player(conv['background_splitted'], player_llm, player_chat_mode)
            l2l_conv = []
            while True:
                l2l_conv.append(h.generate_response(l2l_conv))
                l2l_conv.append(p.generate_response(l2l_conv))
                if h.is_conv_end(l2l_conv) or len(l2l_conv) > 22:
                    break
            for c in l2l_conv:
                print(">>> Next message:\n", c)
                print()
            conv['l2l'][0] = l2l_conv
            print()
            print()
            print()

        # result_idx += 1
        if i == 26 - 1:
            break
    with open(output_path, "w") as json_file:
        json.dump(all_conv, json_file, ensure_ascii=False, indent=2)


def test_helper(task_data_path, provider_constructor, provider_llm):
    all_conv = data_combination(read_path(task_data_path))
    evaluation_set = [i for i in range(31)]
    evaluate_results = []

    for i, one_type in enumerate(all_conv):
        if i not in evaluation_set:
            continue
        evaluate_results.append([])
        for j, conv in enumerate(one_type):

            print("{0}.{1}".format(i+1, j))
            evaluate_results[i].append([])
            

            # if i != 1 - 1 or j != 1:
            #     continue

            print('------------------------------')
            print(conv['background'])
            print('------------------------------')
            print()
            gold_r = conv['all_response'].strip().split('\n')
            h = provider_constructor(gold_r, conv['background_splitted'], conv['gold_structure'], conv, provider_llm)

            l2l_conv = []
            while True:
                l2l_conv.append(h.generate_response(l2l_conv))
                for c in l2l_conv:
                    print(c)
                    print()
                user_input = input("Enter your response: ")  # User inputs the response
                l2l_conv.append(user_input)  # User input is appended instead of p.generate_response

                if h.is_conv_end(l2l_conv) or len(l2l_conv) > 24:
                    break
            print()
            print()
            print()
            exit()




if __name__ == "__main__":
    
    load_dotenv()

    parser = argparse.ArgumentParser(description='Run language model evaluation.')
    parser.add_argument('--seeker_agent_llm', type=str, default='gpt4o')
    parser.add_argument('--provider_agent_llm', type=str, default='gpt4o')
    parser.add_argument('--task_data_path', type=str, default='data/English', help='Path to task data')
    
    parser.add_argument('--player_chat_mode', action='store_true', help='Enable player chat mode')
    parser.add_argument('--multi_info_provider_agent', action='store_true', help='Use a multiple info provider agent instead of a general provider agent.')
    parser.add_argument('--play_around', action='store_true')
    parser.add_argument('--hftoken', type=str, help='Huggingface token if deepseek model is used')
    parser.add_argument('--evaluation_set', type=str, default='0-25',   
                    help='Files to evaluate. Format: "0-25" for range, "0,5,10" for specific files')
    # steering arguments
    parser.add_argument("--steering_feature", type=int, default=None,
                    help="SAE feature index to steer on (None = no steering)")
    parser.add_argument("--steering_strength", type=float, default=1.0,
                        help="Steering strength multiplier")
    parser.add_argument("--steering_max_act", type=float, default=None,
                        help="Optional fixed max activation for the feature")
    parser.add_argument("--compute_max_per_turn", action="store_true",
                        help="If set, estimate max_act per prompt instead of using a fixed value")


    args = parser.parse_args()

    # Parse evaluation_set argument  
    if ',' in args.evaluation_set:  
        # Comma-separated: "0,5,10"  
        evaluation_set = [int(x.strip()) for x in args.evaluation_set.split(',')]  
    elif '-' in args.evaluation_set:  
        # Range: "0-25"  
        start, end = args.evaluation_set.split('-')  
        evaluation_set = [i for i in range(int(start), int(end) + 1)]  
    else:  
        # Single file: "5"  
        evaluation_set = [int(args.evaluation_set)]

    eval_tag = f"set{args.evaluation_set.replace(',', '_').replace('-', '_')}"
    
    player_chat_mode = args.player_chat_mode
    task_data_path = args.task_data_path
    hftoken = args.hftoken
    # print(f"got value for token from function {hftoken}")

    provider_agent_constructor = multi_info_provider if args.multi_info_provider_agent else general_provider
    language = 'En' if task_data_path == 'data/English' else 'Ch'
    mode = 'Chat' if player_chat_mode else 'Comp'

    # Steering metadata (used only for Gemma / HookedGEMMA)
    steering_meta = {
        "steering_feature": getattr(args, "steering_feature", None),
        "steering_strength": getattr(args, "steering_strength", 1.0),
        "steering_max_act": getattr(args, "steering_max_act", 3.0),
    }


    if args.play_around:
        test_helper(task_data_path, provider_agent_constructor, args.provider_agent_llm)
        exit()
    
    if args.seeker_agent_llm == "gpt4o":
        player_llm = ChatGPT("gpt-4o-2024-05-13", 'log/gpt4o_plyaer_cache.pkl')
        output_path = "results/l2l_gpt4o.{}.{}.json".format(mode,language)
    elif args.seeker_agent_llm == 'qianfan':
        player_llm = QianFan("ERNIE-Bot-4", 'log/qianfan_plyaer_cache.pkl')
        output_path = "results/l2l_qianfan.{}.{}.json".format(mode,language)
    elif args.seeker_agent_llm == 'llama':
        if player_chat_mode:
            player_llm = LLAMA("[your-llama-model-path] max_new_tokens:150", 'log/llama2_plyaer_cache.pkl')
        else:
            player_llm = LLAMA("[google/gemma-2b-it] max_new_tokens:150", 'log/llama2_plyaer_cache.pkl')
        output_path = "results/l2l_llama.{}.{}.json".format(mode,language)
    elif args.seeker_agent_llm  == 'llama3.1-405B':
        player_llm = AWSBedrockLLAMA("llama3.1-405B", 'log/llm_player_cache_llama3.1-405B.pkl')
        output_path = "results/l2l_llama3.1-405B.{}.{}.json".format(mode,language)
    elif args.seeker_agent_llm == 'deepseek':
        player_llm = CustomLLM(args.seeker_agent_llm, api_key=hftoken, cache=f'log/llm_player_cache_deepseek.pkl')
        output_path = "results/l2l_deepseek.{}.{}.{}.json".format(mode,language, eval_tag)
    elif args.seeker_agent_llm == 'gemma':
        player_llm = HookedGEMMA(
            model_name="gemma-2b-it",
            sae_release="gemma-2b-it-res-jb",
            sae_id="blocks.12.hook_resid_post",
            device="cuda",
            steering_feature=args.steering_feature,
            steering_strength=args.steering_strength,
            max_act=args.steering_max_act,
            compute_max_per_turn=args.compute_max_per_turn,
        )

        # Suffix for output file
        if steering_meta["steering_feature"] is not None:
            suffix = "steering_f{}_s{}_m{}".format(
                steering_meta["steering_feature"],
                steering_meta["steering_strength"],
                steering_meta["steering_max_act"],
            )
        else:
            suffix = "nosteering"

        output_path = "results/l2l_gemma.{}.{}.{}.{}.json".format(mode, language, eval_tag, suffix)

    else:
        player_llm = HuggingFaceLLM(args.seeker_agent_llm, f'log/{args.seeker_agent_llm.split("/")[-1]}_plyaer_cache.pkl')
        output_path = "results/l2l_{}.{}.{}.{}.json".format(args.seeker_agent_llm.split("/")[-1], mode, language, eval_tag)

    evaluate_player(task_data_path, output_path, player_llm, player_chat_mode, provider_agent_constructor, args.provider_agent_llm, hftoken, evaluation_set)


