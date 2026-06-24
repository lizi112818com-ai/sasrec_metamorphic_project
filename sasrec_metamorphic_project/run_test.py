# -*- coding: utf-8 -*-
"""一键运行完整实验流程。

流程包括：
1. 数据预处理；
2. SASRec 模型训练；
3. 推荐准确率评估；
4. 蜕变测试；
5. 保存实验报告。
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import torch

from metrics import evaluate_model
from metamorphic_test import run_metamorphic_tests
from preprocess import load_processed_data, preprocess
from train import load_model, train_model


def save_accuracy_metrics(metrics: dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "accuracy_metrics.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["模型"] + list(metrics.keys()))
        writer.writeheader()
        writer.writerow({"模型": "SASRec", **{k: round(v, 4) for k, v in metrics.items()}})
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/ml-1m", help="MovieLens-1M 目录，需包含 ratings.dat")
    parser.add_argument("--data_path", type=str, default="data/processed_ml1m.pkl")
    parser.add_argument("--model_path", type=str, default="outputs/sasrec.pt")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--demo", action="store_true", help="使用内置小型示例数据，无需下载 MovieLens")
    parser.add_argument("--force_preprocess", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--maxlen", type=int, default=50)
    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--max_users", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.force_preprocess or not os.path.exists(args.data_path):
        preprocess(
            data_dir=args.data_dir,
            output_path=args.data_path,
            demo=args.demo,
            maxlen=args.maxlen,
            seed=args.seed,
        )

    if args.epochs > 0 or not os.path.exists(args.model_path):
        train_model(
            data_path=args.data_path,
            save_path=args.model_path,
            maxlen=args.maxlen,
            hidden_units=args.hidden_units,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            dropout_rate=args.dropout_rate,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            device=args.device,
            seed=args.seed,
        )

    data = load_processed_data(args.data_path)
    data["maxlen"] = args.maxlen
    model = load_model(args.model_path, device=args.device)

    acc = evaluate_model(model, data, split="test", k=args.k, device=args.device)
    acc_path = save_accuracy_metrics(acc, args.output_dir)
    print(f"[run_test] 推荐准确率结果：{acc}")
    print(f"[run_test] 推荐准确率已保存：{acc_path}")

    summary, failed_cases = run_metamorphic_tests(
        model=model,
        data=data,
        output_dir=args.output_dir,
        k=args.k,
        threshold=args.threshold,
        max_users=args.max_users,
        device=args.device,
        seed=args.seed,
    )
    print("[run_test] 蜕变测试结果：")
    for row in summary:
        print(row)

    report_path = os.path.join(args.output_dir, "final_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "metamorphic_summary": summary, "failed_cases": failed_cases}, f, ensure_ascii=False, indent=2)
    print(f"[run_test] 最终报告已保存：{report_path}")


if __name__ == "__main__":
    main()
