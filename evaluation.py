from ALL_KEYS import *
from utils.data_loader import *
from utils.llm import ChatGPT, AWSBedrockLLAMA, CustomLLM
from utils.utils import detect_language
import sys
import json


def data2prompt_mini(gold, glod_explain, predict):
    """
    Build a prompt asking an evaluator LLM to decide whether the
    generated information (predict) covers the gold information.

    gold: list[str] – gold sentences
    glod_explain: list[str] – explanation / purpose for each gold sentence
    predict: list[str] – model-generated sentences
    """

    def add_punctuation(sentence: str, Chinese: bool = False) -> str:
        if not sentence:
            return sentence
        if sentence[-1] not in ['.', '?', '!', ',', '。', '？', '！', '，']:
            return sentence + ('。' if Chinese else '.')
        return sentence

    if not gold:
        # Degenerate case – return a trivial prompt
        return "Gold information is empty. Return match: true in JSON."

    is_chinese = detect_language(gold[0]) == "Chinese"

    if is_chinese:
        g_name = '正确信息及其信息用途'
        p_name = '模型生成的信息'

        start = (
            f"下面有两段文本。第一段是：{g_name}。第二段是：{p_name}。"
            f"请判断第二段中的信息是否包含了第一段中的每一条信息。"
        )

        numbered_gold = [
            f"{i+1}. {add_punctuation(line, Chinese=True)}\t\t信息用途：{ex}"
            for (i, line), ex in zip(enumerate(gold), glod_explain)
        ]
        numbered_predict = [
            f"{i+1}. {add_punctuation(line, Chinese=True)}"
            for i, line in enumerate(predict)
        ]

        middle = (
            f"{g_name}：\n" + "\n".join(numbered_gold) +
            "\n\n" +
            f"{p_name}：\n" + "\n".join(numbered_predict)
        )

        end = (
            "请仔细判断每一条正确信息是否在生成信息中完整出现或被合理蕴含。"
            "最后请给出一个总体判断：生成信息是否完整覆盖正确信息。"
            "请只用 JSON 格式回答，形如 "
            "{\"analysis content\": \"你的分析\", \"match\": true 或 false}。"
        )

        return "\n\n".join([start, middle, end])

    # English version
    g_name = 'Gold Information and its Purpose'
    p_name = 'Generated Information'

    start = (
        f"Below are two passages of text. The first is: {g_name}. "
        f"The second is: {p_name}. "
        f"Determine whether the {p_name} fully contains or clearly implies "
        f"every piece of {g_name}."
    )

    numbered_gold = [
        f"{i+1}. {add_punctuation(line)}\t\tInformation Purpose: {ex}"
        for (i, line), ex in zip(enumerate(gold), glod_explain)
    ]
    numbered_predict = [
        f"{i+1}. {add_punctuation(line)}"
        for i, line in enumerate(predict)
    ]

    middle = (
        f"{g_name}:\n" + "\n".join(numbered_gold) +
        "\n\n" +
        f"{p_name}:\n" + "\n".join(numbered_predict)
    )

    end = (
        "Carefully determine whether each line from the Gold Information is fully "
        "present in, or clearly implied by, the Generated Information. "
        "Respond in JSON with the fields: "
        "{\"analysis content\": \"...\", \"match\": true or false}."
    )

    return "\n\n".join([start, middle, end])


def evaluate_one_multi(gold, gold_explain, predict, llm):
    """
    Evaluate one multi-step conversation.

    gold: list of gold helper lines (as loaded from JSON, including first dummy line)
    gold_explain: list of explanations for each gold line (same length as gold[1:])
    predict: list of helper lines produced in this run
    llm: evaluator LLM (must support .request(prompt, None, previous_message=None, json_format=True))
    """
    # Strip leading "Jax:" if present and whitespace
    gold_norm = [
        s[4:].strip() if s.lower().startswith("jax:") else s.strip()
        for s in gold[1:]  # skip the first line to match original logic
    ]
    pred_norm = [
        s[4:].strip() if s.lower().startswith("jax:") else s.strip()
        for s in predict
    ]

    if not gold_norm:
        return 0

    # Language-specific lowercasing
    if detect_language(gold_norm[0]) != "Chinese":
        gold_norm = [g.lower() for g in gold_norm]
        pred_norm = [p.lower() for p in pred_norm]

    # Easy exact check
    if gold_norm == pred_norm:
        return 1

    # Easy coverage check: every gold line is somewhere in predict
    if all(g in pred_norm for g in gold_norm):
        return 1

    # Fuzzy coverage: strip trailing punctuation, build coverage sets
    def strip_punct(x: str) -> str:
        return x.rstrip('，。？！,.?!')

    gold_clean = [strip_punct(g) for g in gold_norm]
    pred_clean = [strip_punct(p) for p in pred_norm]

    contained_gold = set()
    for p in pred_clean:
        for g in gold_clean:
            if g and g in p:
                contained_gold.add(g)
                contained_gold.add(p)

    # Map gold_clean -> explanation
    gold_explain_map = {gc: ex for gc, ex in zip(gold_clean, gold_explain)}

    gold_diff = [g for g in gold_clean if g not in contained_gold]
    predict_diff = [p for p in pred_clean if p not in contained_gold]
    glod_explain_diff = [gold_explain_map.get(g, "") for g in gold_diff]

    # If everything is covered, success
    if not gold_diff:
        return 1
    # If no extra predicted info, fail
    if not predict_diff:
        return 0

    last_result = None
    # Ask the evaluator LLM for each uncovered gold item
    for gd, gde in zip(gold_diff, glod_explain_diff):
        prompt = data2prompt_mini([gd], [gde], predict_diff)
        response, _ = llm.request(prompt, None, previous_message=None, json_format=True)
        try:
            result_dict = json.loads(response)
        except Exception:
            result_dict = {}
        last_result = result_dict
        if 'match' in result_dict and not result_dict['match']:
            break

    if last_result and last_result.get('match', False):
        return 1
    return 0


def evaluate_l2l_doc():
    # Arguments:
    #   1: llm_name (evaluator LLM)
    #   2: json_file (results file)
    #   3: api_key  (for evaluator LLM if needed)
    #   4: evaluation_set (e.g. '0-30' or '0,1,5')
    llm_name = sys.argv[1]
    json_file = sys.argv[2]
    api_key = sys.argv[3]
    raw_evaluation_set = sys.argv[4]

    # Parse evaluation_set string
    evaluation_set = set()
    if '-' in raw_evaluation_set:
        start, end = raw_evaluation_set.split('-')
        start, end = int(start), int(end)
        for i in range(start, end + 1):
            evaluation_set.add(i)
    else:
        for part in raw_evaluation_set.split(','):
            part = part.strip()
            if part:
                evaluation_set.add(int(part))

    # Choose evaluator LLM
    if llm_name == 'llama3.1-405b':
        llm = AWSBedrockLLAMA("llama3.1-405b", 'log/llama3.1_evaluator_cache.pkl')
    elif llm_name == 'qwen':
        llm = CustomLLM(
            llm_name,
            api_key=api_key,
            cache=f'log/llm_helpers_cache_{llm_name}.pkl'
        )
    else:
        llm = ChatGPT("gpt-4o-2024-05-13", 'log/llm_evaluator_cache.pkl')

    # Load JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        all_conv = json.load(f)

    meta = {}
    if isinstance(all_conv, dict) and "data" in all_conv:
        meta = all_conv.get("meta", {}) or {}
        all_conv = all_conv["data"]

    # Containers
    evaluate_results = []
    AQD_evaluation_results = []
    ARL_evaluation_results = []

    # metrics for steering
    STEP_recall_results = []      # partial coverage of gold steps
    CLARQ_count_results = []      # # of seeker questions per dialogue
    CLARQ_rate_results = []       # fraction of seeker turns that are questions
    CLARQ_depth_results = []      # last question turn index (1-based)
    GOODBYE_results = []          # 1 if final seeker turn contains "goodbye"
    QLEN_results = []             # avg words per clarifying question

    result_idx = 0

    for i, one_type in enumerate(all_conv):
        if i not in evaluation_set:
            continue

        evaluate_results.append([])
        AQD_evaluation_results.append([])
        ARL_evaluation_results.append([])

        STEP_recall_results.append([])
        CLARQ_count_results.append([])
        CLARQ_rate_results.append([])
        CLARQ_depth_results.append([])
        GOODBYE_results.append([])
        QLEN_results.append([])

        for j, conv in enumerate(one_type):
            evaluate_results[result_idx].append([])
            AQD_evaluation_results[result_idx].append([])
            ARL_evaluation_results[result_idx].append([])

            STEP_recall_results[result_idx].append([])
            CLARQ_count_results[result_idx].append([])
            CLARQ_rate_results[result_idx].append([])
            CLARQ_depth_results[result_idx].append([])
            GOODBYE_results[result_idx].append([])
            QLEN_results[result_idx].append([])

            # Gold helper responses (multi-line string -> list)
            gold_r = conv['all_response'].strip().split('\n')

            for h2l in conv.get('l2l', []):
                if not h2l:
                    continue

                helper_response = []
                seeker_reponse = []

                # h2l[0] is usually the initial Jax message; we start from h2l[1:]
                for k, sent in enumerate(h2l[1:]):
                    if k % 2 == 1 and k != 1:
                        helper_response.append(sent)
                    elif k % 2 == 0:
                        seeker_reponse.append(sent.strip())

                # --- strict success ---
                strict_ok = evaluate_one_multi(
                    gold_r,
                    conv['all_response_exaplain'],
                    helper_response,
                    llm
                )
                evaluate_results[result_idx][j].append(strict_ok)

                # --- AQD (Average Query Discrepancy) ---
                AQD_evaluation_results[result_idx][j].append(
                    len(helper_response) + 1 - len(gold_r)
                )

                # --- ARL (Average Query Length) ---
                if seeker_reponse:
                    if detect_language(seeker_reponse[0]) != "Chinese":
                        ARL_evaluation_results[result_idx][j].append(
                            sum(s.count(' ') for s in seeker_reponse) / len(seeker_reponse)
                        )
                    else:
                        ARL_evaluation_results[result_idx][j].append(
                            sum(len(s) for s in seeker_reponse) / len(seeker_reponse)
                        )
                else:
                    ARL_evaluation_results[result_idx][j].append(0.0)

                # 1) Step Recall: fraction of gold steps covered (simple substring coverage)
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

                STEP_recall_results[result_idx][j].append(step_recall)

                # 2) ClarQ_count & ClarQ_rate: how many questions the seeker asks
                num_seeker_turns = len(seeker_reponse)
                num_questions = sum('?' in s for s in seeker_reponse)
                question_rate = num_questions / num_seeker_turns if num_seeker_turns > 0 else 0.0

                CLARQ_count_results[result_idx][j].append(num_questions)
                CLARQ_rate_results[result_idx][j].append(question_rate)

                # 3) ClarQ_depth: last question turn index (1-based, 0 if none)
                last_q_idx = -1
                for idx, s in enumerate(seeker_reponse):
                    if '?' in s:
                        last_q_idx = idx
                depth = last_q_idx + 1 if last_q_idx >= 0 else 0
                CLARQ_depth_results[result_idx][j].append(depth)

                # 4) Goodbye compliance: does the final seeker turn contain "goodbye"?
                goodbye_flag = 0
                if seeker_reponse:
                    last_msg = seeker_reponse[-1].lower()
                    if "goodbye" in last_msg:
                        goodbye_flag = 1
                GOODBYE_results[result_idx][j].append(goodbye_flag)

                # 5) Average clarifying question length (words) per dialogue
                question_lengths = [s.count(' ') + 1 for s in seeker_reponse if '?' in s]
                if question_lengths:
                    avg_q_len = sum(question_lengths) / len(question_lengths)
                else:
                    avg_q_len = 0.0
                QLEN_results[result_idx][j].append(avg_q_len)

        result_idx += 1

    # Helper to aggregate a 3-level nested list [type][conv][run]
    def sum_nested(arr):
        return sum(sum(sum(inner) for inner in outer) for outer in arr)

    denom = 10 * len(evaluation_set) if evaluation_set else 1

    # Original metrics
    sum_arrays = sum_nested(evaluate_results)
    print("Success Rate: {}".format(sum_arrays / denom))

    sum_arrays = sum_nested(AQD_evaluation_results)
    print("Average Query Discrepancy (AQD): {}".format(sum_arrays / denom))

    sum_arrays = sum_nested(ARL_evaluation_results)
    print("Average Query Length (ARL): {}".format(sum_arrays / denom))

    # New metrics
    sum_arrays = sum_nested(STEP_recall_results)
    print("Average Step Recall: {}".format(sum_arrays / denom))

    sum_arrays = sum_nested(CLARQ_count_results)
    print("Average # Questions per Dialogue (ClarQ_count): {}".format(sum_arrays / denom))

    sum_arrays = sum_nested(CLARQ_rate_results)
    print("Average Question Rate (ClarQ_rate): {}".format(sum_arrays / denom))

    sum_arrays = sum_nested(CLARQ_depth_results)
    print("Average Clarification Depth (ClarQ_depth): {}".format(sum_arrays / denom))

    sum_arrays = sum_nested(GOODBYE_results)
    print("Goodbye Compliance Rate: {}".format(sum_arrays / denom))

    sum_arrays = sum_nested(QLEN_results)
    print("Average Clarifying Question Length (words): {}".format(sum_arrays / denom))


if __name__ == "__main__":
    evaluate_l2l_doc()
