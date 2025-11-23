import json
import os
from typing import List, Dict, Any

import pandas as pd

from utils.utils import detect_language
from utils.llm import ChatGPT, AWSBedrockLLAMA, CustomLLM


def data2prompt_mini(gold, glod_explain, predict):
    def add_punctuation(sentence, Chinese=False):
        if sentence and sentence[-1] not in ['.', '?', '!', ',', '。', '？', '！', '，']:
            return sentence + ('。' if Chinese else '.')
        return sentence

    if not gold:
        return "Gold information is empty. Return match: true in JSON."

    if detect_language(gold[0]) == "Chinese":
        g_name = '正确信息及其信息用途'
        p_name = '生成信息'
        start = (
            "下面展示了两段文字。第一段是：{0}，第二段是：{1}。第一段的信息用途可以帮助你理解正确信息的用途。"
            "{1}缺少这些用途的说明，你需要自行分析{1}的用途，并判断其是否包含了正确信息的用途。"
            "\n判断方法是先分析{1}的用途，然后检查是否可以在{1}中找到与正确信息用途一致的内容。"
        ).format(g_name, p_name)

        numbered_gold = [
            f"{i+1}. {add_punctuation(line, True)} 信息用途：{add_punctuation(line_ex, True)}"
            for (i, line), line_ex in zip(enumerate(gold), glod_explain)
        ]
        numbered_predict = [f"{i+1}. {add_punctuation(line, True)}" for i, line in enumerate(predict)]

        middle = f"{g_name}：\n" + "\n".join(numbered_gold) + f"\n\n{p_name}：\n" + "\n".join(numbered_predict)

        end = (
            "仔细判断第一段是否被第二段提及。返回格式为："
            "{ 'analysis': '...', 'match': true/false }"
        )
        return "\n\n".join([start, middle, end])

    g_name = 'Gold Information and its Purpose'
    p_name = 'Generated Information'

    start = (
        "Below are two passages of text. First: {0}, second: {1}. Analyze whether the second contains "
        "the purpose of the gold information."
    ).format(g_name, p_name)

    numbered_gold = [
        f"{i+1}. {add_punctuation(line)}  Purpose: {add_punctuation(line_ex)}"
        for (i, line), line_ex in zip(enumerate(gold), glod_explain)
    ]
    numbered_predict = [f"{i+1}. {add_punctuation(line)}" for i, line in enumerate(predict)]

    middle = f"{g_name}:\n" + "\n".join(numbered_gold) + f"\n\n{p_name}:\n" + "\n".join(numbered_predict)

    end = (
        "Return JSON with fields { 'analysis': '...', 'match': true/false }."
    )
    return "\n\n".join([start, middle, end])


def evaluate_one_multi(gold, gold_explain, predict, llm) -> int:
    import json as _json

    gold = [s[4:].strip() if s.lower().startswith("jax:") else s.strip()
            for s in gold[1:]]
    predict = [s[4:].strip() if s.lower().startswith("jax:") else s.strip()
               for s in predict]

    if not gold:
        return 0

    # checks whether matches directly
    if detect_language(gold[0]) != "Chinese":
        gold_norm = [g.lower() for g in gold]
        pred_norm = [p.lower() for p in predict]
    else:
        gold_norm = gold[:]
        pred_norm = predict[:]

    if gold_norm == pred_norm:
        return 1

    # checks whether all gold contained
    if all(g in pred_norm for g in gold_norm):
        return 1

    # if partial match then LLM judge
    gold_clean = [g.rstrip('，。？！,.?!') for g in gold_norm]
    pred_clean = [p.rstrip('，。？！,.?!') for p in pred_norm]

    contained = set()
    explain_map = {g: gold_explain[i] for i, g in enumerate(gold_clean)}

    for p in pred_clean:
        for g in gold_clean:
            if g in p:
                contained.add(g)

    gold_diff = [g for g in gold_clean if g not in contained]
    pred_diff = [p for p in pred_clean if p not in contained]

    if not gold_diff:
        return 1

    if not pred_diff:
        return 0

    for gd in gold_diff:
        gde = explain_map[gd]
        prompt = data2prompt_mini([gd], [gde], pred_diff)

        response, _ = llm.request(prompt, None, previous_message=None, json_format=True)
        try:
            parsed = _json.loads(response)
        except Exception:
            return 0

        if not parsed.get("match", False):
            return 0

    return 1


def build_llm(llm_name: str, api_key: str):
    if llm_name == 'llama3.1-405b':
        return AWSBedrockLLAMA("llama3.1-405b", 'log/llama3.1_evaluator_cache.pkl')
    elif llm_name == 'qwen':
        return CustomLLM(llm_name, api_key=api_key, cache=f'log/llm_helpers_cache_{llm_name}.pkl')
    else:
        # default judge: GPT-4o
        return ChatGPT("gpt-4o-2024-05-13", 'log/llm_evaluator_cache.pkl')


def compute_metrics_for_file(json_file: str,
                             llm,
                             evaluation_set: List[int]) -> Dict[str, Any]:

    with open(json_file, "r", encoding="utf-8") as f:
        all_conv = json.load(f)

    success_sum = aqd_sum = arl_sum = step_sum = 0.0
    cq_count_sum = cq_rate_sum = cq_depth_sum = goodbye_sum = 0.0
    qlen_sum = 0.0

    denom = 10 * len(evaluation_set)

    for i, one_type in enumerate(all_conv):
        if i not in evaluation_set:
            continue

        for conv in one_type:
            gold_r = conv["all_response"].split("\n")

            for h2l in conv.get("l2l", []):
                if not h2l:
                    continue

                helper = []
                seeker = []

                # Split helper/seeker turns
                for k, sent in enumerate(h2l[1:]):
                    if k % 2 == 1 and k != 1:
                        helper.append(sent)
                    elif k % 2 == 0:
                        seeker.append(sent.strip())

                # Success rate
                strict = evaluate_one_multi(
                    gold_r,
                    conv["all_response_exaplain"],
                    helper,
                    llm
                )
                success_sum += strict

                # AQD
                aqd_sum += (len(helper) + 1 - len(gold_r))

                # ARL
                if seeker:
                    if detect_language(seeker[0]) != "Chinese":
                        arl_sum += sum(s.count(" ") for s in seeker) / len(seeker)
                    else:
                        arl_sum += sum(len(s) for s in seeker) / len(seeker)
                else:
                    arl_sum += 0

                # Step recall
                gold_norm = [s[4:].strip() if s.startswith("Jax:") else s.strip()
                             for s in gold_r[1:]]
                pred_norm = [s[4:].strip() if s.startswith("Jax:") else s.strip()
                             for s in helper]

                if detect_language(gold_norm[0]) != "Chinese":
                    gold_norm = [g.lower() for g in gold_norm]
                    pred_norm = [p.lower() for p in pred_norm]

                def strip_p(x): return x.rstrip("，。？！,.?!")

                gold_clean = [strip_p(g) for g in gold_norm]
                pred_clean = [strip_p(p) for p in pred_norm]

                covered = sum(any(g in p for p in pred_clean) for g in gold_clean)
                step_sum += covered / len(gold_clean) if gold_clean else 0.0

                # Clarifying Questions
                num_turns = len(seeker)
                num_q = sum("?" in s for s in seeker)

                cq_count_sum += num_q
                cq_rate_sum += (num_q / num_turns) if num_turns > 0 else 0.0

                # Depth
                last_q = -1
                for idx, s in enumerate(seeker):
                    if "?" in s:
                        last_q = idx
                cq_depth_sum += (last_q + 1) if last_q >= 0 else 0

                # goodbye
                goodbye_sum += 1 if (seeker and "goodbye" in seeker[-1].lower()) else 0

                # Q-length
                lengths = [s.count(" ") + 1 for s in seeker if "?" in s]
                qlen_sum += (sum(lengths) / len(lengths)) if lengths else 0.0

    return {
        "file": os.path.basename(json_file),
        "success_rate": success_sum / denom,
        "AQD": aqd_sum / denom,
        "ARL": arl_sum / denom,
        "step_recall": step_sum / denom,
        "ClarQ_count": cq_count_sum / denom,
        "ClarQ_rate": cq_rate_sum / denom,
        "ClarQ_depth": cq_depth_sum / denom,
        "Goodbye_rate": goodbye_sum / denom,
        "ClarQ_len": qlen_sum / denom,
    }


def evaluate_files(
    json_paths: List[str],
    llm_name: str = "qwen",
    api_key: str = "",
    evaluation_set: str = "3",
) -> pd.DataFrame:
    """
    Example usage:
        df = evaluate_files(
            ["file1.json", "file2.json"],
            llm_name="qwen",
            api_key=API_KEY,
            evaluation_set="3",
        )
    """

    # Convert "3", "0-5", "0,2,4" to list[int]
    def parse_eval(raw: str):
        raw = raw.strip()
        if "," in raw:
            return [int(x) for x in raw.split(",") if x]
        if "-" in raw:
            a, b = raw.split("-")
            return list(range(int(a), int(b) + 1))
        return [int(raw)]

    eval_set_list = parse_eval(evaluation_set)

    # Build judge LLM
    llm = build_llm(llm_name, api_key)

    # Evaluate all files
    rows = []
    for path in json_paths:
        print(f"Evaluating {path} ...")
        metrics = compute_metrics_for_file(path, llm, eval_set_list)
        rows.append(metrics)

    df = pd.DataFrame(rows)
    return df
