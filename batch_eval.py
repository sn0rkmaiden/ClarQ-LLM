import argparse
import json
import os
from typing import Dict, Any, List

import pandas as pd

from utils.utils import detect_language
from utils.llm import ChatGPT, AWSBedrockLLAMA, CustomLLM


def data2prompt_mini(gold, glod_explain, predict):
    """
    Same semantics as in your evaluation.py:
    Build a prompt asking an evaluator LLM to decide whether the
    generated information (predict) covers the gold information.
    """
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
            "{1}缺少这些用途的说明，你需要自行分析{1}的用途，并判断其是否包含了正确信息的用途。\n"
            "判断方法是先分析{1}的用途，然后检查是否可以在{1}中找到与正确信息用途一致的内容。如果正确信息的用途能在{1}中的某一条信息中找到相同的对应项，我们就认为比对成功。"
            "注意：正确信息通常用于澄清特定内容，如果{1}只是列举了多个可能的场景、物品、人物、技能等名称，那么{1}可能需要依赖正确信息进行澄清，从而可能导致不匹配的情况。"
        ).format(g_name, p_name)

        numbered_gold = [
            f'{i+1}. {add_punctuation(line, True)}\t\t\t\t信息用途：{add_punctuation(line_ex, True)}'
            for (i, line), line_ex in zip(enumerate(gold), glod_explain)
        ]
        numbered_predict = [
            f'{i+1}. {add_punctuation(line, True)}'
            for i, line in enumerate(predict)
        ]

        middle = '{0}：\n{1}\n\n{2}：\n{3}'.format(
            g_name, '\n'.join(numbered_gold), p_name, '\n'.join(numbered_predict)
        )
        end = (
            "仔细判断第一段{0}的每一行是否都在第二段{1}中被提及？即检查是否可以在{1}中找到与正确信息用途一致的内容。如果正确信息的用途能在{1}中的某一条信息中找到相同的对应项，我们就认为比对成功。"
            "注意：正确信息通常用于澄清特定内容，如果{1}只是列举了多个可能的场景、物品、人物、技能等名称，那么{1}可能需要依赖正确信息进行澄清，从而可能导致不匹配的情况。\n"
            "返回包含两个字段的JSON对象：一个'analysis'字段，它的值是你对此的分析字符串，以及一个'match'字段，他的值是布尔类型，当比对成功时为True。格式应如下所示："
            "{{ 'analysis': '您的分析内容', 'match': True 或 False }}。"
        ).format(g_name, p_name)

        return '\n\n'.join([start, middle, end])

    # English
    g_name = 'Gold Information and its Purpose'
    p_name = 'Generated Information'
    start = (
        "Below are two passages of text. The first is: {0}, and the second is: {1}. "
        "The purpose of the information in the first passage can help you understand the gold information. "
        "Since {1} lacks an explanation of these purposes, you need to analyze the purpose of {1} on your own and determine whether it includes the purpose of the gold information.\n"
        "The method is to first analyze the purpose of {1}, then check whether you can find content in {1} that matches the purpose of the gold information. "
        "If the purpose of the gold information can be found in any part of {1}, we consider the comparison successful. "
        "Note: Gold information is typically used to clarify specific content. If {1} merely lists multiple possible scenes, objects, characters, or skill names, "
        "the {1} may need to rely on gold information for clarification, which could lead to mismatches."
    ).format(g_name, p_name)

    numbered_gold = [
        f'{i+1}. {add_punctuation(line)}\t\t\t\tPurpose of Information: {add_punctuation(line_ex)}'
        for (i, line), line_ex in zip(enumerate(gold), glod_explain)
    ]
    numbered_predict = [
        f'{i+1}. {add_punctuation(line)}'
        for i, line in enumerate(predict)
    ]

    middle = '{0}：\n{1}\n\n{2}：\n{3}'.format(
        g_name, '\n'.join(numbered_gold), p_name, '\n'.join(numbered_predict)
    )

    end = (
        "Carefully determine whether each line from the {0} passage is mentioned in the {1} passage. "
        "Check whether you can find content in {1} that matches the purpose of the gold information. "
        "If the purpose of the gold information can be found in any part of {1}, we consider the comparison successful. "
        "Note: Gold information is typically used to clarify specific content. If {1} merely lists multiple possible scenes, objects, characters, or skill names, "
        "the {1} may need to rely on gold information for clarification, which could lead to mismatches.\n"
        "Return a JSON object containing two fields: an 'analysis' field, whose value is your analytical string about this, and a 'match' field, which is a Boolean indicating whether comparison was successful. "
        "The format should be as follows: {{ 'analysis': 'Your analysis content', 'match': True or False }}."
    ).format(g_name, p_name)

    return '\n\n'.join([start, middle, end])


def evaluate_one_multi(gold, gold_explain, predict, llm) -> int:
    """
    Same logic as your evaluation.py:
    returns 1 if helper coverage is judged sufficient, else 0.
    """
    import json as _json

    gold = [s[4:].strip() if s.lower().startswith("jax:") else s.strip() for s in gold[1:]]
    predict = [s[4:].strip() if s.lower().startswith("jax:") else s.strip() for s in predict]

    if not gold:
        return 0

    easy_check = (gold == predict)
    if detect_language(gold[0]) != "Chinese":
        gold = [g.lower() for g in gold]
        predict = [g.lower() for g in predict]

    if easy_check:
        return 1

    # all gold in predict?
    easy_check = True
    for g in gold:
        if g not in predict:
            easy_check = False
            break
    if easy_check:
        return 1

    gold_cleaned = [sentence.rstrip('，。？！,.?!') for sentence in gold]
    predict_cleaned = [sentence.rstrip('，。？！,.?!') for sentence in predict]

    contained_gold = set()
    contained_gold_explain = {}
    for p in predict_cleaned:
        for g in gold_cleaned:
            if g in p:
                contained_gold.add(g)
                contained_gold.add(p)
    for i_g, g in enumerate(gold_cleaned):
        contained_gold_explain[g] = gold_explain[i_g]

    gold_diff = [g for g in gold_cleaned if g not in contained_gold]
    predict_diff = [p for p in predict_cleaned if p not in contained_gold]
    glod_explain_diff = [contained_gold_explain[g] for g in gold_diff]

    if not gold_diff:
        return 1
    elif not predict_diff:
        return 0

    result_dict = {}
    for gd, gde in zip(gold_diff, glod_explain_diff):
        prompt = data2prompt_mini([gd], [gde], predict_diff)
        resonse, _ = llm.request(prompt, None, previous_message=None, json_format=True)
        try:
            result_dict = _json.loads(resonse)
        except Exception:
            # if parsing fails, treat as non-match
            return 0
        if 'match' in result_dict and not result_dict['match']:
            break

    if 'match' in result_dict and result_dict['match']:
        return 1
    return 0


def parse_evaluation_set(raw: str) -> List[int]:
    """
    Same semantics as old evaluation.py:
    - '3'         -> [3]
    - '0-5'       -> [0,1,2,3,4,5]
    - '0,2,4'     -> [0,2,4]
    """
    raw = raw.strip()
    if ',' in raw:
        return [int(x.strip()) for x in raw.split(',') if x.strip()]
    if '-' in raw:
        start, end = raw.split('-')
        return [i for i in range(int(start), int(end) + 1)]
    return [int(raw)]


def build_llm(llm_name: str, api_key: str):
    if llm_name == 'llama3.1-405b':
        return AWSBedrockLLAMA("llama3.1-405b", 'log/llama3.1_evaluator_cache.pkl')
    elif llm_name == 'qwen':
        return CustomLLM(llm_name, api_key=api_key, cache=f'log/llm_helpers_cache_{llm_name}.pkl')
    else:
        # default: gpt-4o
        return ChatGPT("gpt-4o-2024-05-13", 'log/llm_evaluator_cache.pkl')


def compute_metrics_for_file(
    json_file: str,
    llm,
    evaluation_set: List[int],
) -> Dict[str, Any]:
    """
    Compute all metrics for a single ClarQ-LLM result JSON.
    Returns a flat dict with aggregated metrics.
    """

    with open(json_file, 'r', encoding='utf-8') as f:
        all_conv = json.load(f)

    # Accumulators (sum over types / convs / runs)
    success_sum = 0.0
    aqd_sum = 0.0
    arl_sum = 0.0
    step_recall_sum = 0.0
    clarq_count_sum = 0.0
    clarq_rate_sum = 0.0
    clarq_depth_sum = 0.0
    goodbye_sum = 0.0
    qlen_sum = 0.0

    # We keep the same denominator convention as original evaluation.py:
    # 10 conversations per type * number of types evaluated.
    denom = 10 * len(evaluation_set) if evaluation_set else 1

    for i, one_type in enumerate(all_conv):
        if i not in evaluation_set:
            continue

        for j, conv in enumerate(one_type):
            gold_r = conv['all_response'].strip().split('\n')

            for h2l in conv.get('l2l', []):
                if not h2l:
                    continue

                helper_response = []
                seeker_reponse = []
                for k, sent in enumerate(h2l[1:]):
                    if k % 2 == 1 and k != 1:
                        helper_response.append(sent)
                    elif k % 2 == 0:
                        seeker_reponse.append(sent.strip())

                # 1) Success Rate
                strict_ok = evaluate_one_multi(
                    gold_r,
                    conv['all_response_exaplain'],
                    helper_response,
                    llm
                )
                success_sum += strict_ok

                # 2) AQD
                aqd_sum += (len(helper_response) + 1 - len(gold_r))

                # 3) ARL (Average Query Length)
                if seeker_reponse:
                    if detect_language(seeker_reponse[0]) != "Chinese":
                        arl_sum += sum(s.count(' ') for s in seeker_reponse) / len(seeker_reponse)
                    else:
                        arl_sum += sum(len(s) for s in seeker_reponse) / len(seeker_reponse)
                else:
                    arl_sum += 0.0

                # 4) Step Recall (simple substring-based)
                gold_norm = [
                    s[4:].strip() if s.lower().startswith("jax:") else s.strip()
                    for s in gold_r[1:]
                ]
                pred_norm = [
                    s[4:].strip() if s.lower().startswith("jax:") else s.strip()
                    for s in helper_response
                ]

                step_recall = 0.0
                if gold_norm:
                    if detect_language(gold_norm[0]) != "Chinese":
                        gold_norm = [g.lower() for g in gold_norm]
                        pred_norm = [p.lower() for p in pred_norm]

                    def strip_punct2(x: str) -> str:
                        return x.rstrip('，。？！,.?!')

                    gold_clean2 = [strip_punct2(g) for g in gold_norm]
                    pred_clean2 = [strip_punct2(p) for p in pred_norm]

                    covered = 0
                    for g in gold_clean2:
                        if any(g and g in p for p in pred_clean2):
                            covered += 1
                    step_recall = covered / len(gold_clean2)

                step_recall_sum += step_recall

                # 5) ClarQ_count & ClarQ_rate
                num_seeker_turns = len(seeker_reponse)
                num_questions = sum('?' in s for s in seeker_reponse)
                question_rate = num_questions / num_seeker_turns if num_seeker_turns > 0 else 0.0

                clarq_count_sum += num_questions
                clarq_rate_sum += question_rate

                # 6) ClarQ_depth (1-based index of last question turn)
                last_q_idx = -1
                for idx, s in enumerate(seeker_reponse):
                    if '?' in s:
                        last_q_idx = idx
                depth = last_q_idx + 1 if last_q_idx >= 0 else 0
                clarq_depth_sum += depth

                # 7) Goodbye compliance
                goodbye_flag = 0
                if seeker_reponse:
                    last_msg = seeker_reponse[-1].lower()
                    if "goodbye" in last_msg:
                        goodbye_flag = 1
                goodbye_sum += goodbye_flag

                # 8) Avg clarifying question length (words)
                question_lengths = [s.count(' ') + 1 for s in seeker_reponse if '?' in s]
                if question_lengths:
                    avg_q_len = sum(question_lengths) / len(question_lengths)
                else:
                    avg_q_len = 0.0
                qlen_sum += avg_q_len

    # Normalize by same denominator as evaluation.py
    metrics = {
        "file": os.path.basename(json_file),
        "success_rate": success_sum / denom,
        "AQD": aqd_sum / denom,
        "ARL": arl_sum / denom,
        "step_recall": step_recall_sum / denom,
        "ClarQ_count": clarq_count_sum / denom,
        "ClarQ_rate": clarq_rate_sum / denom,
        "ClarQ_depth": clarq_depth_sum / denom,
        "Goodbye_rate": goodbye_sum / denom,
        "ClarQ_len": qlen_sum / denom,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm_name",
        type=str,
        default="qwen",
        help="Name of evaluator LLM: 'qwen', 'llama3.1-405b', or 'gpt4o' (default: qwen)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="API key for evaluator LLM (used for qwen / OpenAI etc.)",
    )
    parser.add_argument(
        "--evaluation_set",
        type=str,
        required=True,
        help="Which scenario indices to evaluate, e.g. '3', '0-5', '0,2,4'",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional path to save results as CSV",
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        help="Paths to ClarQ-LLM result JSON files",
    )

    args = parser.parse_args()

    evaluation_set = parse_evaluation_set(args.evaluation_set)
    llm = build_llm(args.llm_name, args.api_key)

    rows: List[Dict[str, Any]] = []
    for jf in args.json_files:
        print(f"Evaluating {jf} ...")
        metrics = compute_metrics_for_file(jf, llm, evaluation_set)
        rows.append(metrics)

    df = pd.DataFrame(rows)
    print("\n=== Aggregated metrics ===")
    print(df)

    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"\nSaved to {args.output_csv}")


if __name__ == "__main__":
    main()
