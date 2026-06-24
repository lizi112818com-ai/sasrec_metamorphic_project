# -*- coding: utf-8 -*-
"""面向序列推荐的蜕变测试模块。"""

from __future__ import annotations

import csv
import json
import os
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from metrics import jaccard_at_k, rank_difference_at_k, recommend_topk


def noise_insertion(seq: Sequence[int], item_num: int, seed: int | None = None) -> List[int]:
    """噪声插入：向序列中插入一个用户未交互物品。"""
    rng = random.Random(seed)
    seq = list(seq)
    used = set(seq)
    noise = rng.randint(1, item_num)
    tries = 0
    while noise in used and tries < 1000:
        noise = rng.randint(1, item_num)
        tries += 1
    if len(seq) <= 1:
        return seq + [noise]
    # 不插入最后一个位置之后，避免完全改变“最近行为”的定义。
    pos = rng.randint(0, max(0, len(seq) - 2))
    return seq[: pos + 1] + [noise] + seq[pos + 1 :]


def sequence_truncation(seq: Sequence[int], keep_ratio: float = 0.6) -> List[int]:
    """序列截断：删除较早历史，仅保留最近 keep_ratio 比例的行为。"""
    seq = list(seq)
    keep_len = max(2, int(np.ceil(len(seq) * keep_ratio)))
    return seq[-keep_len:]


def similar_item_replacement(seq: Sequence[int], model, item_num: int, seed: int | None = None, device: str = "cpu") -> List[int]:
    """相似物品替换：基于 SASRec 的 item embedding 寻找相似物品。"""
    rng = random.Random(seed)
    seq = list(seq)
    if len(seq) == 0:
        return seq
    # 避免替换最后一个物品，尽量保持近期兴趣。
    pos = rng.randint(0, max(0, len(seq) - 2)) if len(seq) > 1 else 0
    target = seq[pos]

    with torch.no_grad():
        emb = model.item_emb.weight.detach().to(device)  # [item_num+1, H]
        target_emb = emb[target]
        norm_emb = torch.nn.functional.normalize(emb[1:], dim=-1)
        target_norm = torch.nn.functional.normalize(target_emb.unsqueeze(0), dim=-1)
        sims = torch.matmul(norm_emb, target_norm.t()).squeeze(-1).detach().cpu().numpy()

    # 从相似度高到低选择一个非自身、非序列内物品。
    order = np.argsort(-sims) + 1
    used = set(seq)
    replacement = target
    for item in order:
        item = int(item)
        if item != target and item not in used:
            replacement = item
            break
    new_seq = seq.copy()
    new_seq[pos] = replacement
    return new_seq


def order_perturbation(seq: Sequence[int], seed: int | None = None) -> List[int]:
    """顺序扰动：交换较早位置的相邻两个物品。"""
    rng = random.Random(seed)
    seq = list(seq)
    if len(seq) < 2:
        return seq
    upper = max(0, len(seq) // 2 - 1)
    pos = rng.randint(0, upper) if upper > 0 else 0
    new_seq = seq.copy()
    new_seq[pos], new_seq[pos + 1] = new_seq[pos + 1], new_seq[pos]
    return new_seq


def transform_sequence(rule: str, seq: Sequence[int], model, item_num: int, seed: int | None = None, device: str = "cpu") -> List[int]:
    if rule == "噪声插入":
        return noise_insertion(seq, item_num, seed=seed)
    if rule == "序列截断":
        return sequence_truncation(seq, keep_ratio=0.6)
    if rule == "相似替换":
        return similar_item_replacement(seq, model, item_num, seed=seed, device=device)
    if rule == "顺序扰动":
        return order_perturbation(seq, seed=seed)
    raise ValueError(f"未知蜕变关系：{rule}")


@torch.no_grad()
def run_metamorphic_tests(
    model,
    data: Dict,
    output_dir: str = "outputs",
    k: int = 10,
    threshold: float = 0.3,
    max_users: int | None = None,
    device: str = "cpu",
    seed: int = 2024,
) -> Tuple[List[Dict], List[Dict]]:
    """执行四类蜕变关系测试，并返回汇总结果与失败样例。"""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    item_num = data["itemnum"]
    maxlen = int(data.get("maxlen", 50))
    user_train = data["user_train"]
    user_valid = data["user_valid"]

    users = list(user_train.keys())
    if max_users is not None:
        users = users[:max_users]

    rules = ["噪声插入", "序列截断", "相似替换", "顺序扰动"]
    summary: List[Dict] = []
    failed_cases: List[Dict] = []

    for rule in rules:
        jac_values, rd_values = [], []
        violation_count, total_count = 0, 0
        for idx, u in enumerate(users):
            # 使用训练序列 + 验证物品作为测试时的历史行为。
            seq = list(user_train[u]) + [int(user_valid[u])]
            if len(seq) < 3:
                continue
            mut_seq = transform_sequence(rule, seq, model, item_num, seed=seed + idx, device=device)

            # 推荐时屏蔽对应序列中已经交互过的物品。
            top_orig = recommend_topk(model, seq, maxlen=maxlen, k=k, device=device, mask_items=set(seq))
            top_mut = recommend_topk(model, mut_seq, maxlen=maxlen, k=k, device=device, mask_items=set(mut_seq))

            jac = jaccard_at_k(top_orig, top_mut, k=k)
            rd = rank_difference_at_k(top_orig, top_mut, k=k)
            jac_values.append(jac)
            rd_values.append(rd)
            total_count += 1

            if jac < threshold:
                violation_count += 1
                if len(failed_cases) < 30:
                    failed_cases.append(
                        {
                            "user": int(u),
                            "rule": rule,
                            "original_sequence": seq[-10:],
                            "mutated_sequence": mut_seq[-10:],
                            "topk_original": top_orig,
                            "topk_mutated": top_mut,
                            f"Jaccard@{k}": round(float(jac), 4),
                            f"RankDifference@{k}": round(float(rd), 4),
                        }
                    )

        result = {
            "测试规则": rule,
            f"Jac@{k}": round(float(np.mean(jac_values)), 4) if jac_values else 0.0,
            f"RD@{k}": round(float(np.mean(rd_values)), 4) if rd_values else 0.0,
            "VR": f"{(violation_count / total_count * 100):.1f}%" if total_count else "0.0%",
            "测试样本数": total_count,
        }
        summary.append(result)

    # 保存 CSV 和 JSON 报告。
    summary_path = os.path.join(output_dir, "metamorphic_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    failed_path = os.path.join(output_dir, "failed_cases.json")
    with open(failed_path, "w", encoding="utf-8") as f:
        json.dump(failed_cases, f, ensure_ascii=False, indent=2)

    print(f"[metamorphic_test] 汇总结果已保存：{summary_path}")
    print(f"[metamorphic_test] 失败样例已保存：{failed_path}")
    return summary, failed_cases
