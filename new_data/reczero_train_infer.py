#!/usr/bin/env python3
"""RecZero training + inference pipeline adapted to `new_data` format.

Data assumptions (tab-separated):
- <PREFIX>_user_items_negs_train.csv: user_id<TAB>pos_item_seq_csv<TAB>neg_item_pool_csv
- <PREFIX>_user_items_negs_test.csv:  user_id<TAB>pos_item_seq_csv<TAB>neg_item_pool_csv
- <PREFIX>_item_desc.tsv: headers include `item_id` and `summary`

Inference evaluates ranking among: 1 target + 1000 sampled negatives.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


@dataclass
class UserSample:
    user_id: int
    history: List[int]
    target: int
    neg_pool: List[int]


def parse_user_items_file(path: Path) -> Dict[int, Tuple[List[int], List[int]]]:
    records: Dict[int, Tuple[List[int], List[int]]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row_idx, row in enumerate(reader, start=1):
            if len(row) < 3:
                print(f"[Data][WARN] {path.name} line {row_idx}: expected 3 cols, got {len(row)}")
                continue
            user_id = int(row[0])
            pos = [int(x) for x in row[1].split(",") if x != ""]
            neg = [int(x) for x in row[2].split(",") if x != ""]
            records[user_id] = (pos, neg)
    return records


def parse_item_summary(path: Path) -> Dict[int, str]:
    out: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            item_id = row.get("item_id")
            if item_id is None or item_id == "":
                continue
            sid = int(item_id)
            summary = (row.get("summary") or "").strip()
            out[sid] = summary if summary else f"item_{sid}"
    return out


def build_samples(records: Dict[int, Tuple[List[int], List[int]]]) -> List[UserSample]:
    samples: List[UserSample] = []
    for user_id, (pos, neg) in records.items():
        if len(pos) < 2:
            continue
        samples.append(UserSample(user_id=user_id, history=pos[:-1], target=pos[-1], neg_pool=neg))
    return samples


class MeanHistoryRanker(nn.Module):
    def __init__(self, item_vectors: torch.Tensor, trainable_item_adapter_dim: int = 256):
        super().__init__()
        self.register_buffer("base_item_vectors", item_vectors)
        in_dim = item_vectors.size(1)
        self.user_proj = nn.Linear(in_dim, trainable_item_adapter_dim)
        self.item_proj = nn.Linear(in_dim, trainable_item_adapter_dim)

    def encode_history(self, histories: Sequence[Sequence[int]]) -> torch.Tensor:
        reps = []
        for h in histories:
            h_t = torch.tensor(h, dtype=torch.long, device=self.base_item_vectors.device)
            rep = self.base_item_vectors.index_select(0, h_t).mean(dim=0)
            reps.append(rep)
        hist = torch.stack(reps, dim=0)
        return F.normalize(self.user_proj(hist), dim=-1)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        base = self.base_item_vectors.index_select(0, item_ids)
        return F.normalize(self.item_proj(base), dim=-1)


def encode_item_texts(
    model_name: str,
    item_texts: Dict[int, str],
    num_items: int,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    print(f"[Backbone] Loading tokenizer/model from Hugging Face: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    hidden_size = model.config.hidden_size
    embeddings = torch.zeros((num_items, hidden_size), dtype=torch.float32, device=device)

    valid_item_ids = sorted(item_texts.keys())
    total = len(valid_item_ids)
    print(f"[Backbone] Encoding {total} item summaries into fixed item vectors")
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_ids = valid_item_ids[start:end]
        texts = [item_texts[i] for i in batch_ids]
        inputs = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**inputs)
            vec = out.last_hidden_state[:, 0, :]
            vec = F.normalize(vec, dim=-1)
        embeddings[torch.tensor(batch_ids, device=device)] = vec
        print(f"[Backbone][{end}/{total}] Encoded item vectors")
    return embeddings


def sample_negative(neg_pool: List[int], all_items: List[int], banned: set[int], rng: random.Random) -> int:
    valid_from_pool = [x for x in neg_pool if x not in banned]
    if valid_from_pool:
        return rng.choice(valid_from_pool)
    while True:
        x = rng.choice(all_items)
        if x not in banned:
            return x


def sample_eval_negatives(
    neg_pool: List[int],
    all_items: List[int],
    banned: set[int],
    k: int,
    rng: random.Random,
) -> List[int]:
    candidates = [x for x in neg_pool if x not in banned]
    rng.shuffle(candidates)
    selected = []
    used = set()
    for x in candidates:
        if x in used:
            continue
        selected.append(x)
        used.add(x)
        if len(selected) == k:
            return selected

    while len(selected) < k:
        x = rng.choice(all_items)
        if x in banned or x in used:
            continue
        selected.append(x)
        used.add(x)
    return selected


def check_data_leakage(train: List[UserSample], test: List[UserSample]) -> None:
    train_users = {x.user_id for x in train}
    test_users = {x.user_id for x in test}
    overlap = train_users & test_users
    if overlap:
        raise ValueError(f"[Leakage] train/test user overlap detected: {len(overlap)} users")
    print(f"[Leakage] PASS user split disjoint: train={len(train_users)} test={len(test_users)}")


def run_training(
    model: MeanHistoryRanker,
    train_samples: List[UserSample],
    all_items: List[int],
    epochs: int,
    lr: float,
    seed: int,
) -> None:
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    print(f"[Train] Start: epochs={epochs}, samples={len(train_samples)}, lr={lr}")
    for ep in range(1, epochs + 1):
        rng.shuffle(train_samples)
        running = 0.0
        for idx, s in enumerate(train_samples, start=1):
            banned = set(s.history)
            banned.add(s.target)
            neg = sample_negative(s.neg_pool, all_items, banned, rng)

            user_vec = model.encode_history([s.history])
            cand_ids = torch.tensor([s.target, neg], dtype=torch.long, device=user_vec.device)
            item_vec = model.encode_items(cand_ids)
            logits = (user_vec @ item_vec.T) / 0.07
            labels = torch.zeros(1, dtype=torch.long, device=user_vec.device)
            loss = F.cross_entropy(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += float(loss.item())
            print(
                f"[Train][Epoch {ep}/{epochs}][Step {idx}/{len(train_samples)}] "
                f"user={s.user_id} target={s.target} neg={neg} loss={loss.item():.6f}"
            )

        print(f"[Train][Epoch {ep}] avg_loss={running / max(1, len(train_samples)):.6f}")


def hit_and_ndcg(rank_pos: int, k: int) -> Tuple[float, float]:
    if rank_pos <= k:
        return 1.0, 1.0 / math.log2(rank_pos + 1)
    return 0.0, 0.0


def run_inference(
    model: MeanHistoryRanker,
    test_samples: List[UserSample],
    all_items: List[int],
    neg_k: int,
    seed: int,
) -> Dict[str, float]:
    rng = random.Random(seed + 999)
    model.eval()

    sum_hr10 = sum_hr20 = sum_hr40 = 0.0
    sum_ndcg10 = sum_ndcg20 = sum_ndcg40 = 0.0
    sum_rank = 0.0

    total = len(test_samples)
    print(f"[Infer] Start: users={total}, candidates_per_user={neg_k + 1}")

    with torch.no_grad():
        for idx, s in enumerate(test_samples, start=1):
            banned = set(s.history)
            banned.add(s.target)
            negs = sample_eval_negatives(s.neg_pool, all_items, banned, k=neg_k, rng=rng)
            cand = [s.target] + negs

            user_vec = model.encode_history([s.history])
            item_ids = torch.tensor(cand, dtype=torch.long, device=user_vec.device)
            item_vec = model.encode_items(item_ids)
            scores = (user_vec @ item_vec.T).squeeze(0)
            sorted_idx = torch.argsort(scores, descending=True)
            target_rank = int((sorted_idx == 0).nonzero(as_tuple=False).item()) + 1

            hr10, nd10 = hit_and_ndcg(target_rank, 10)
            hr20, nd20 = hit_and_ndcg(target_rank, 20)
            hr40, nd40 = hit_and_ndcg(target_rank, 40)
            sum_hr10 += hr10
            sum_hr20 += hr20
            sum_hr40 += hr40
            sum_ndcg10 += nd10
            sum_ndcg20 += nd20
            sum_ndcg40 += nd40
            sum_rank += target_rank

            n = idx
            print(
                f"[Infer][User {idx}/{total}] user={s.user_id} target_rank={target_rank} | "
                f"Avg HR@10={sum_hr10/n:.4f}, HR@20={sum_hr20/n:.4f}, HR@40={sum_hr40/n:.4f} | "
                f"Avg NDCG@10={sum_ndcg10/n:.4f}, NDCG@20={sum_ndcg20/n:.4f}, NDCG@40={sum_ndcg40/n:.4f}, "
                f"Avg Rank={sum_rank/n:.2f}"
            )

    n = max(1, total)
    return {
        "hr@10": sum_hr10 / n,
        "hr@20": sum_hr20 / n,
        "hr@40": sum_hr40 / n,
        "ndcg@10": sum_ndcg10 / n,
        "ndcg@20": sum_ndcg20 / n,
        "ndcg@40": sum_ndcg40 / n,
        "avg_rank": sum_rank / n,
        "users": total,
    }


def save_metrics(metrics: Dict[str, float], out_path: Path) -> None:
    lines = [f"{k}\t{v}" for k, v in metrics.items()]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Output] metrics saved to {out_path}")


def detect_paths(data_dir: Path, dataset_prefix: str) -> Tuple[Path, Path, Path]:
    train = data_dir / f"{dataset_prefix}_user_items_negs_train.csv"
    test = data_dir / f"{dataset_prefix}_user_items_negs_test.csv"
    item = data_dir / f"{dataset_prefix}_item_desc.tsv"
    if not train.exists() or not test.exists() or not item.exists():
        raise FileNotFoundError(
            f"Missing required files for prefix={dataset_prefix}:\n"
            f"  - {train}\n  - {test}\n  - {item}"
        )
    return train, test, item


def main() -> None:
    parser = argparse.ArgumentParser(description="RecZero pipeline for new_data format")
    parser.add_argument("--data-dir", default="new_data")
    parser.add_argument("--dataset-prefix", default="Baby_Products")
    parser.add_argument("--hf-backbone", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--eval-neg-k", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save-dir", default="outputs/reczero_new_data")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--max-train-users", type=int, default=0)
    parser.add_argument("--max-test-users", type=int, default=0)
    args = parser.parse_args()

    t0 = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("[Step 1/8] Resolving input files")
    train_path, test_path, item_path = detect_paths(data_dir, args.dataset_prefix)
    print(f"[Data] train={train_path.name} | test={test_path.name} | item={item_path.name}")

    print("[Step 2/8] Loading train/test user files")
    train_records = parse_user_items_file(train_path)
    test_records = parse_user_items_file(test_path)

    print("[Step 3/8] Building train/test samples")
    train_samples = build_samples(train_records)
    test_samples = build_samples(test_records)

    if args.max_train_users > 0:
        train_samples = train_samples[: args.max_train_users]
    if args.max_test_users > 0:
        test_samples = test_samples[: args.max_test_users]

    print(f"[Data] usable train samples={len(train_samples)}, test samples={len(test_samples)}")

    print("[Step 4/8] Loading item summaries")
    item_texts = parse_item_summary(item_path)
    max_item_id = max(item_texts.keys()) if item_texts else 0
    num_items = max_item_id + 1
    all_items = list(range(num_items))
    print(f"[Data] items with summary={len(item_texts)} total_item_ids={num_items}")

    print("[Step 5/8] Data leakage checks")
    check_data_leakage(train_samples, test_samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Runtime] device={device}")

    print("[Step 6/8] Download/load HF backbone and encode item vectors")
    item_vectors = encode_item_texts(
        model_name=args.hf_backbone,
        item_texts=item_texts,
        num_items=num_items,
        batch_size=args.encode_batch_size,
        max_length=args.max_length,
        device=device,
    )

    print("[Step 7/8] Training recommender")
    model = MeanHistoryRanker(item_vectors=item_vectors).to(device)
    run_training(
        model=model,
        train_samples=train_samples,
        all_items=all_items,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )

    print("[Step 8/8] Inference + metrics over 1 target + 1000 random negatives")
    metrics = run_inference(
        model=model,
        test_samples=test_samples,
        all_items=all_items,
        neg_k=args.eval_neg_k,
        seed=args.seed,
    )

    metrics_path = save_dir / f"{args.dataset_prefix}_metrics.tsv"
    save_metrics(metrics, metrics_path)

    if args.save_model:
        model_path = save_dir / f"{args.dataset_prefix}_adapter.pt"
        torch.save(model.state_dict(), model_path)
        print(f"[Output] fine-tuned adapter saved to {model_path}")

    print("[Done] Summary metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  - {k}: {v:.6f}")
        else:
            print(f"  - {k}: {v}")
    print(f"[Done] total elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
