# 基于 SASRec 的序列推荐蜕变测试项目

本项目对应课程论文《基于蜕变测试的序列推荐模型鲁棒性分析与测试实践》的代码实现部分，包含数据预处理、SASRec 模型训练、蜕变测试和结果分析四个部分。

## 1. 文件说明

```text
sasrec_metamorphic_project/
├── preprocess.py          # 读取 MovieLens-1M，构造按时间排序的用户行为序列
├── model.py               # SASRec 模型：物品嵌入、位置嵌入、因果自注意力层、预测层
├── train.py               # 负采样、模型训练、验证集选择、模型保存
├── metamorphic_test.py    # 噪声插入、序列截断、相似替换、顺序扰动四类蜕变测试
├── metrics.py             # HR、Recall、NDCG、Jaccard、Rank Difference、Violation Rate
├── run_test.py            # 一键组织完整实验流程并保存测试报告
├── requirements.txt       # 依赖环境
├── data/                  # 数据目录
└── outputs/               # 模型、指标、失败样例和最终报告输出目录
```

## 2. 安装依赖

建议使用 Python 3.9 及以上版本。

```bash
pip install -r requirements.txt
```

## 3. 快速运行：使用内置演示数据

若暂时没有 MovieLens-1M 数据，可以直接运行：

```bash
python run_test.py --demo --force_preprocess --epochs 1 --maxlen 10 --hidden_units 16 --max_users 10
```

运行后会在 `outputs/` 目录下生成：

```text
outputs/sasrec.pt                 # 训练后的 SASRec 模型
outputs/accuracy_metrics.csv      # 推荐准确率结果
outputs/metamorphic_summary.csv   # 四类蜕变关系下的鲁棒性结果
outputs/failed_cases.json         # 失败样例
outputs/final_report.json         # 最终整合报告
```

## 4. 使用 MovieLens-1M 数据运行

请下载 MovieLens-1M 数据集，并将 `ratings.dat` 放到：

```text
data/ml-1m/ratings.dat
```

然后执行：

```bash
python run_test.py --force_preprocess --epochs 50 --batch_size 128
```

如果已经预处理和训练过，只想重新执行测试：

```bash
python run_test.py --epochs 0
```

## 5. 分步运行方式

### 5.1 数据预处理

```bash
python preprocess.py --data_dir data/ml-1m --output data/processed_ml1m.pkl
```

### 5.2 训练 SASRec

```bash
python train.py --data_path data/processed_ml1m.pkl --save_path outputs/sasrec.pt --epochs 50
```

### 5.3 执行完整评估与蜕变测试

```bash
python run_test.py --data_path data/processed_ml1m.pkl --model_path outputs/sasrec.pt --epochs 0
```

## 6. 论文中可写的代码实现描述

本文代码实现包括数据预处理、模型训练、蜕变测试和结果分析四个部分。`preprocess.py` 读取 MovieLens-1M 数据，并生成按时间排序的用户行为序列；`model.py` 实现 SASRec 模型，包括物品嵌入、位置嵌入、因果自注意力层和预测层；`train.py` 负责负采样、模型训练和验证集选择；`metamorphic_test.py` 负责生成四类变异序列并调用模型推理；`metrics.py` 实现 Recall@K、NDCG@K、HitRate@K、Jaccard@K、Rank Difference@K 和 Violation Rate；`run_test.py` 负责组织完整实验流程并保存测试报告。

## 7. 注意事项

1. 课程论文中的实验表格应优先使用你实际运行得到的 `accuracy_metrics.csv` 和 `metamorphic_summary.csv`。
2. 若使用 `--demo`，结果只适合验证代码流程，不适合作为正式实验结果。
3. 若在 CPU 上训练 MovieLens-1M，建议先减少 `--epochs` 或设置 `--max_users` 调试流程。
