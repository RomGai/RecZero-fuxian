# RecZero new_data 训练+推理流程

该流程会：
1. 自动根据文件名加载 train/test 用户集合（`*_train.csv` vs `*_test.csv`）。
2. 从 Hugging Face 下载并加载 backbone（默认 `sentence-transformers/all-MiniLM-L6-v2`），编码 item summary。
3. 训练一个轻量 ranker（用户历史均值向量 + 可训练投影层）。
4. 推理时对每个用户执行 **1个target + 1000个随机负样本** 的排序。
5. 输出并实时打印累计平均指标：`HR@10/20/40`、`NDCG@10/20/40`、`Avg Rank`。

## 直接可运行命令（按顺序）

### 0) （可选）检查参数
```bash
python new_data/reczero_train_infer.py --help
```

### 1) Baby_Products 全流程
```bash
python new_data/reczero_train_infer.py \
  --data-dir new_data \
  --dataset-prefix Baby_Products \
  --hf-backbone sentence-transformers/all-MiniLM-L6-v2 \
  --epochs 2 \
  --eval-neg-k 1000 \
  --save-dir outputs/reczero_new_data \
  --save-model
```

### 2) Video_Games 全流程
```bash
python new_data/reczero_train_infer.py \
  --data-dir new_data \
  --dataset-prefix Video_Games \
  --hf-backbone sentence-transformers/all-MiniLM-L6-v2 \
  --epochs 2 \
  --eval-neg-k 1000 \
  --save-dir outputs/reczero_new_data \
  --save-model
```

## 快速 smoke test（小规模）
```bash
python new_data/reczero_train_infer.py \
  --dataset-prefix Baby_Products \
  --max-train-users 64 \
  --max-test-users 64 \
  --epochs 1 \
  --eval-neg-k 1000
```
