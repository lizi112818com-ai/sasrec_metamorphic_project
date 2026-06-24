# -*- coding: utf-8 -*-
"""推荐准确率指标与蜕变测试稳定性指标。"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


def hitrate_at_k(rank: int, k: int) -> float:
    return 1.0 if 1 <= rank <= k else 0.0


def recall_at_k(rank: int, k: int) -> float:
    # leave-one-out 中每个用户只有一个测试物品，因此 Recall@K 与 HR@K 相同。
    return 1.0 if 1 <= rank <= k else 0.0


def ndcg_at_k(rank: int, k: int) -> float:
    if 1 <= rank <= k:
        return 1.0 / math.log2(rank + 1)
    return 0.0


def jaccard_at_k(list_a: Sequence[int], list_b: Sequence[int], k: int = 10) -> float:
    set_a = set(list_a[:k])
    set_b = set(list_b[:k])
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def rank_difference_at_k(list_a: Sequence[int], list_b: Sequence[int], k: int = 10) -> float:
    top_a = list(list_a[:k])
    top_b = list(list_b[:k])
    common = set(top_a) & set(top_b)
    if not common:
        return float(k)
    rank_a = {item: idx + 1 for idx, item in enumerate(top_a)}
    rank_b = {item: idx + 1 for idx, item in enumerate(top_b)}
    diffs = [abs(rank_a[item] - rank_b[item]) for item in common]
    return float(np.mean(diffs))


def pad_sequence(seq: Sequence[int], maxlen: int) -> List[int]:
    seq = list(seq)[-maxlen:]
    return [0] * (maxlen - len(seq)) + seq


@torch.no_grad()
def recommend_topk(
    model,
    seq: Sequence[int],
    maxlen: int,
    k: int = 10,
    device: str = "cpu",
    mask_items: Iterable[int] | None = None,
) -> List[int]:
    model.eval()
    arr = torch.LongTensor([pad_sequence(seq, maxlen)]).to(device)
    scores = model.predict(arr)[0].detach().cpu().numpy()  # 对物品 1..item_num 打分
    if mask_items is not None:
        for item in mask_items:
            if 1 <= int(item) <= model.item_num:
                scores[int(item) - 1] = -1e9
    top_idx = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return (top_idx + 1).astype(int).tolist()


@torch.no_grad()
def evaluate_model(model, data: Dict, split: str = "test", k: int = 10, device: str = "cpu", batch_size: int = 256) -> Dict[str, float]:
    """评估 SASRec 的 HR@K、Recall@K 和 NDCG@K。

    采用批量推理，并屏蔽用户历史已交互物品。
    """
    model.eval()
    maxlen = int(data.get("maxlen", 50))
    user_train = data["user_train"]
    user_valid = data["user_valid"]
    user_test = data["user_test"]

    examples = []
    for u, train_seq in user_train.items():
        if split == "valid":
            seq = train_seq
            gt = user_valid[u]
        else:
            seq = train_seq + [user_valid[u]]
            gt = user_test[u]
        examples.append((seq, int(gt), set(seq)))

    hrs, recalls, ndcgs = [], [], []
    for st in range(0, len(examples), batch_size):
        batch = examples[st: st + batch_size]
        arr = torch.LongTensor([pad_sequence(x[0], maxlen) for x in batch]).to(device)
        scores = model.predict(arr).detach().cpu().numpy()
        for row, (seq, gt, mask_items) in enumerate(batch):
            for item in mask_items:
                if 1 <= int(item) <= model.item_num:
                    scores[row, int(item) - 1] = -1e9
            gt_score = scores[row, int(gt) - 1]
            rank = int(np.sum(scores[row] > gt_score) + 1)
            hrs.append(hitrate_at_k(rank, k))
            recalls.append(recall_at_k(rank, k))
            ndcgs.append(ndcg_at_k(rank, k))

    return {
        f"HR@{k}": float(np.mean(hrs)) if hrs else 0.0,
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }
