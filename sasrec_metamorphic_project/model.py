# -*- coding: utf-8 -*-
"""SASRec 模型实现。

SASRec 版本：物品嵌入 + 位置嵌入 + 因果自注意力层 + 前馈网络 + 预测层。
该实现保留了 SASRec 的核心思想，即只利用当前位置及其之前的交互行为建模用户兴趣。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CausalSelfAttentionBlock(nn.Module):
    """轻量级因果自注意力模块。"""

    def __init__(self, hidden_units: int, dropout_rate: float = 0.2) -> None:
        super().__init__()
        self.hidden_units = hidden_units
        self.q = nn.Linear(hidden_units, hidden_units)
        self.k = nn.Linear(hidden_units, hidden_units)
        self.v = nn.Linear(hidden_units, hidden_units)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(hidden_units, hidden_units)
        self.norm1 = nn.LayerNorm(hidden_units)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_units, hidden_units * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units * 4, hidden_units),
            nn.Dropout(dropout_rate),
        )
        self.norm2 = nn.LayerNorm(hidden_units)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, L, H], padding_mask: [B, L], True 表示 padding。
        bsz, seq_len, hidden = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scores = torch.bmm(q, k.transpose(1, 2)) / (hidden ** 0.5)  # [B, L, L]

        # 因果掩码：禁止当前位置看到未来位置。
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0), -1e9)
        scores = scores.masked_fill(padding_mask.unsqueeze(1), -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        context = torch.bmm(attn, v)
        x = self.norm1(x + self.out(context))
        x = self.norm2(x + self.ffn(x))
        x = x * (~padding_mask).unsqueeze(-1).float()
        return x


class SASRec(nn.Module):
    """SASRec 序列推荐模型。"""

    def __init__(
        self,
        item_num: int,
        maxlen: int = 50,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,  # 保留参数，便于与论文和命令行保持一致；本轻量版未显式拆分多头。
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.item_num = item_num
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.num_heads = num_heads

        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList([
            CausalSelfAttentionBlock(hidden_units, dropout_rate) for _ in range(num_blocks)
        ])
        self.layer_norm = nn.LayerNorm(hidden_units)

        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(self, log_seqs: torch.Tensor) -> torch.Tensor:
        device = log_seqs.device
        batch_size, seq_len = log_seqs.shape
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.item_emb(log_seqs) + self.pos_emb(positions)
        x = self.emb_dropout(x)
        padding_mask = log_seqs.eq(0)
        for block in self.blocks:
            x = block(x, padding_mask)
        x = self.layer_norm(x)
        x = x * (~padding_mask).unsqueeze(-1).float()
        return x

    def predict(self, log_seqs: torch.Tensor, candidates: Optional[torch.Tensor] = None) -> torch.Tensor:
        """根据用户序列预测候选物品得分。

        candidates=None 时，对全部物品 1..item_num 打分。
        """
        hidden = self.forward(log_seqs)
        final_state = hidden[:, -1, :]
        if candidates is None:
            item_ids = torch.arange(1, self.item_num + 1, device=log_seqs.device)
            item_embs = self.item_emb(item_ids)
            return final_state @ item_embs.t()
        item_embs = self.item_emb(candidates)
        return torch.sum(final_state.unsqueeze(1) * item_embs, dim=-1)
