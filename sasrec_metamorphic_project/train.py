# -*- coding: utf-8 -*-
"""SASRec 训练脚本。"""

from __future__ import annotations

import argparse
import os
import pickle
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from metrics import evaluate_model
from model import SASRec
from preprocess import load_processed_data


def set_seed(seed: int = 2024) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_neg(l: int, r: int, forbidden: set[int]) -> int:
    t = random.randint(l, r)
    while t in forbidden:
        t = random.randint(l, r)
    return t


class SASRecTrainDataset(Dataset):
    def __init__(self, user_train: Dict[int, List[int]], item_num: int, maxlen: int = 50) -> None:
        self.user_train = user_train
        self.users = list(user_train.keys())
        self.item_num = item_num
        self.maxlen = maxlen

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        u = self.users[idx]
        seq = self.user_train[u]
        user_items = set(seq)

        input_seq = np.zeros([self.maxlen], dtype=np.int64)
        pos_seq = np.zeros([self.maxlen], dtype=np.int64)
        neg_seq = np.zeros([self.maxlen], dtype=np.int64)

        nxt = seq[-1]
        idx_pos = self.maxlen - 1
        for item in reversed(seq[:-1]):
            input_seq[idx_pos] = item
            pos_seq[idx_pos] = nxt
            if nxt != 0:
                neg_seq[idx_pos] = random_neg(1, self.item_num, user_items)
            nxt = item
            idx_pos -= 1
            if idx_pos < 0:
                break

        return torch.LongTensor(input_seq), torch.LongTensor(pos_seq), torch.LongTensor(neg_seq)


def train_model(
    data_path: str,
    save_path: str,
    maxlen: int = 50,
    hidden_units: int = 64,
    num_blocks: int = 2,
    num_heads: int = 2,
    dropout_rate: float = 0.2,
    batch_size: int = 128,
    lr: float = 0.001,
    epochs: int = 10,
    device: str = "cpu",
    seed: int = 2024,
) -> SASRec:
    set_seed(seed)
    data = load_processed_data(data_path)
    data["maxlen"] = maxlen
    item_num = data["itemnum"]

    model = SASRec(
        item_num=item_num,
        maxlen=maxlen,
        hidden_units=hidden_units,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
    ).to(device)

    dataset = SASRecTrainDataset(data["user_train"], item_num=item_num, maxlen=maxlen)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    best_ndcg = -1.0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for seq, pos, neg in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
            seq, pos, neg = seq.to(device), pos.to(device), neg.to(device)
            hidden = model(seq)
            pos_emb = model.item_emb(pos)
            neg_emb = model.item_emb(neg)
            pos_logits = (hidden * pos_emb).sum(dim=-1)
            neg_logits = (hidden * neg_emb).sum(dim=-1)
            valid_mask = pos.ne(0)
            loss = bce(pos_logits[valid_mask], torch.ones_like(pos_logits[valid_mask]))
            loss += bce(neg_logits[valid_mask], torch.zeros_like(neg_logits[valid_mask]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        valid_metrics = evaluate_model(model, data, split="valid", k=10, device=device)
        mean_loss = float(np.mean(losses)) if losses else 0.0
        print(f"[train] epoch={epoch}, loss={mean_loss:.4f}, valid={valid_metrics}")
        if valid_metrics["NDCG@10"] > best_ndcg:
            best_ndcg = valid_metrics["NDCG@10"]
            torch.save({"model_state_dict": model.state_dict(), "config": {
                "item_num": item_num,
                "maxlen": maxlen,
                "hidden_units": hidden_units,
                "num_blocks": num_blocks,
                "num_heads": num_heads,
                "dropout_rate": dropout_rate,
            }}, save_path)
            print(f"[train] 保存当前最优模型：{save_path}")

    return model


def load_model(model_path: str, device: str = "cpu") -> SASRec:
    checkpoint = torch.load(model_path, map_location=device)
    cfg = checkpoint["config"]
    model = SASRec(**cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/processed_ml1m.pkl")
    parser.add_argument("--save_path", type=str, default="outputs/sasrec.pt")
    parser.add_argument("--maxlen", type=int, default=50)
    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    train_model(**vars(args))


if __name__ == "__main__":
    main()
