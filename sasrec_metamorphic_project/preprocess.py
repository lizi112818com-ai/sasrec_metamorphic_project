# -*- coding: utf-8 -*-
"""MovieLens-1M 数据预处理脚本。

功能：
1. 读取 MovieLens-1M 的 ratings.dat；
2. 按用户分组并按时间戳排序；
3. 将原始用户编号和物品编号映射为连续整数编号；
4. 按 leave-one-out 方式划分训练集、验证集和测试集；
5. 保存 processed_data.pkl，供 train.py 和 metamorphic_test.py 使用。
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def set_seed(seed: int = 2024) -> None:
    random.seed(seed)
    np.random.seed(seed)


def read_movielens_1m(data_dir: str, min_rating: float = 4.0) -> pd.DataFrame:
    """读取 MovieLens-1M 的 ratings.dat 文件。

    MovieLens-1M 原始格式为：UserID::MovieID::Rating::Timestamp
    """
    rating_path = os.path.join(data_dir, "ratings.dat")
    if not os.path.exists(rating_path):
        raise FileNotFoundError(
            f"未找到 {rating_path}。请将 MovieLens-1M 的 ratings.dat 放入该目录，"
            "或在 run_test.py 中使用 --demo 运行示例数据。"
        )
    df = pd.read_csv(
        rating_path,
        sep="::",
        engine="python",
        names=["user", "item", "rating", "timestamp"],
        encoding="latin-1",
    )
    df = df[df["rating"] >= min_rating].copy()
    df = df.sort_values(["user", "timestamp"])
    return df


def build_sequences(df: pd.DataFrame, min_len: int = 5) -> Tuple[Dict[int, List[int]], int, int, Dict[int, int], Dict[int, int]]:
    """构造用户行为序列，并将用户与物品重新映射为连续编号。"""
    user_ids = sorted(df["user"].unique().tolist())
    item_ids = sorted(df["item"].unique().tolist())
    user2id = {u: idx + 1 for idx, u in enumerate(user_ids)}
    item2id = {i: idx + 1 for idx, i in enumerate(item_ids)}

    df["user_id"] = df["user"].map(user2id)
    df["item_id"] = df["item"].map(item2id)

    user_sequences: Dict[int, List[int]] = {}
    for u, group in df.groupby("user_id"):
        seq = group.sort_values("timestamp")["item_id"].tolist()
        if len(seq) >= min_len:
            user_sequences[int(u)] = seq

    return user_sequences, len(user2id), len(item2id), user2id, item2id


def split_leave_one_out(user_sequences: Dict[int, List[int]]) -> Tuple[Dict[int, List[int]], Dict[int, int], Dict[int, int]]:
    """每个用户最后一个物品作为测试，倒数第二个物品作为验证。"""
    user_train: Dict[int, List[int]] = {}
    user_valid: Dict[int, int] = {}
    user_test: Dict[int, int] = {}
    for u, seq in user_sequences.items():
        if len(seq) < 3:
            continue
        user_train[u] = seq[:-2]
        user_valid[u] = seq[-2]
        user_test[u] = seq[-1]
    return user_train, user_valid, user_test


def generate_demo_dataframe(num_users: int = 12, num_items: int = 50, min_len: int = 6, max_len: int = 12) -> pd.DataFrame:
    """生成一个可运行的演示数据集，便于无 MovieLens 数据时测试项目流程。"""
    rows = []
    timestamp = 1_000_000
    for u in range(1, num_users + 1):
        # 给每个用户分配一个兴趣簇，使序列具有一定结构，而不是完全随机。
        cluster_start = ((u % 5) * 10) + 1
        cluster_items = list(range(cluster_start, min(cluster_start + 10, num_items + 1)))
        seq_len = random.randint(min_len, max_len)
        seq = []
        for _ in range(seq_len):
            if random.random() < 0.75:
                item = random.choice(cluster_items)
            else:
                item = random.randint(1, num_items)
            seq.append(item)
        # 去掉连续重复项，模拟用户交互序列。
        cleaned = []
        for it in seq:
            if not cleaned or cleaned[-1] != it:
                cleaned.append(it)
        for it in cleaned:
            rows.append([u, it, 5, timestamp])
            timestamp += random.randint(1, 10)
    return pd.DataFrame(rows, columns=["user", "item", "rating", "timestamp"])


def preprocess(
    data_dir: str,
    output_path: str,
    min_rating: float = 4.0,
    min_len: int = 5,
    maxlen: int = 50,
    demo: bool = False,
    seed: int = 2024,
) -> str:
    set_seed(seed)
    if demo:
        df = generate_demo_dataframe()
    else:
        df = read_movielens_1m(data_dir, min_rating=min_rating)

    user_sequences, usernum, itemnum, user2id, item2id = build_sequences(df, min_len=min_len)
    user_train, user_valid, user_test = split_leave_one_out(user_sequences)

    id2item = {v: k for k, v in item2id.items()}
    id2user = {v: k for k, v in user2id.items()}

    data = {
        "user_train": user_train,
        "user_valid": user_valid,
        "user_test": user_test,
        "user_sequences": user_sequences,
        "usernum": usernum,
        "itemnum": itemnum,
        "user2id": user2id,
        "item2id": item2id,
        "id2user": id2user,
        "id2item": id2item,
        "maxlen": maxlen,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[preprocess] 保存完成：{output_path}")
    print(f"[preprocess] 用户数：{len(user_train)}，物品数：{itemnum}")
    return output_path


def load_processed_data(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/ml-1m")
    parser.add_argument("--output", type=str, default="data/processed_ml1m.pkl")
    parser.add_argument("--min_rating", type=float, default=4.0)
    parser.add_argument("--min_len", type=int, default=5)
    parser.add_argument("--maxlen", type=int, default=50)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    preprocess(
        data_dir=args.data_dir,
        output_path=args.output,
        min_rating=args.min_rating,
        min_len=args.min_len,
        maxlen=args.maxlen,
        demo=args.demo,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
