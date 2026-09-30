"""Phase 0 (blocking gate): retrain the shallow/medium configurations.

Tests the paper's working theory directly: that attention/FFN slicing damages
pretrained weights that were never trained to tolerate it, and that letting a
configuration adapt (bias + LayerNorm fine-tuning, Stage 6 of Section III-F)
can close some of that gap without touching the shared backbone or the deep
configuration.

Trains on `data/calibration_prompts.json` (799 prompts), a 90/10 train/val
split by prompt index (seed 42). This corpus is disjoint from the 60-prompt
`short_qa` evaluation set used everywhere else in `results/`: calibration
prompts have no `answer` field and are drawn from different categories
(seed_handwritten, basic_arithmetic, ..., hard_reasoning) than the
hand-curated short-QA list, so no held-out contamination is possible.

    python -m calibration.finetune_adapters
    python -m calibration.finetune_adapters --epochs 3 --lr 3e-4
"""

import argparse
import json
import os
import time

import torch

from configs.configurations import CONFIGURATIONS, DEPTH_MAP
from models.adapters import ConfigAdapter, save_adapter
from models.adaptive_model import AdaptiveGPT2
from models.backbone import load_model

PROMPTS_PATH = "data/calibration_prompts.json"
OUTPUT_DIR = "results/phase0_retraining"
VAL_FRACTION = 0.10


def load_split(seed=42):
    with open(PROMPTS_PATH, encoding="utf-8") as handle:
        prompts = [row["text"] for row in json.load(handle)]

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(prompts), generator=generator).tolist()

    n_val = max(1, int(len(prompts) * VAL_FRACTION))
    val_idx = set(order[:n_val])

    train = [prompts[i] for i in order if i not in val_idx]
    val = [prompts[i] for i in order if i in val_idx]
    return train, val


def next_token_loss(logits, input_ids):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )


@torch.no_grad()
def evaluate_split(model, tokenizer, adapter, depth, attention_mode, ffn_mode, prompts):
    model.eval()
    losses = []
    for text in prompts:
        input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=256)["input_ids"]
        if input_ids.shape[1] < 2:
            continue
        logits = model(input_ids, depth=depth, attention_mode=attention_mode,
                       ffn_mode=ffn_mode, adapter=adapter)
        losses.append(next_token_loss(logits, input_ids).item())
    return sum(losses) / len(losses) if losses else float("nan")


def finetune_config(config_name, model, tokenizer, train_prompts, val_prompts,
                    epochs, lr, seed, log_every=50):
    torch.manual_seed(seed)

    spec = CONFIGURATIONS[config_name]
    depth = DEPTH_MAP[config_name]
    attention_mode, ffn_mode = spec["attention_mode"], spec["ffn_mode"]

    adaptive = AdaptiveGPT2(model)
    adapter = ConfigAdapter(model, depth)

    for param in adaptive.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr)

    curve = []
    step = 0
    started = time.perf_counter()

    pre_train_loss = evaluate_split(adaptive, tokenizer, adapter, depth, attention_mode,
                                    ffn_mode, train_prompts[:80])
    pre_val_loss = evaluate_split(adaptive, tokenizer, adapter, depth, attention_mode,
                                  ffn_mode, val_prompts)
    print(f"  [{config_name}] pre-training: train~={pre_train_loss:.4f} val={pre_val_loss:.4f}")
    curve.append({"step": 0, "epoch": 0, "split": "train_sample", "loss": pre_train_loss})
    curve.append({"step": 0, "epoch": 0, "split": "val", "loss": pre_val_loss})

    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(train_prompts),
                               generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        adapter.train()
        running = []

        for i in order:
            text = train_prompts[i]
            input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                                  max_length=256)["input_ids"]
            if input_ids.shape[1] < 2:
                continue

            logits = adaptive(input_ids, depth=depth, attention_mode=attention_mode,
                              ffn_mode=ffn_mode, adapter=adapter)
            loss = next_token_loss(logits, input_ids)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            optimizer.step()

            running.append(loss.item())
            step += 1

            if step % log_every == 0:
                mean_loss = sum(running[-log_every:]) / len(running[-log_every:])
                curve.append({"step": step, "epoch": epoch, "split": "train", "loss": mean_loss})
                print(f"  [{config_name}] epoch {epoch} step {step}/{len(train_prompts)*epochs} "
                      f"train_loss={mean_loss:.4f}", flush=True)

        val_loss = evaluate_split(adaptive, tokenizer, adapter, depth, attention_mode,
                                  ffn_mode, val_prompts)
        curve.append({"step": step, "epoch": epoch, "split": "val", "loss": val_loss})
        print(f"  [{config_name}] epoch {epoch} complete: val_loss={val_loss:.4f} "
              f"({time.perf_counter()-started:.1f}s elapsed)")

    post_train_loss = evaluate_split(adaptive, tokenizer, adapter, depth, attention_mode,
                                     ffn_mode, train_prompts[:80])
    post_val_loss = evaluate_split(adaptive, tokenizer, adapter, depth, attention_mode,
                                   ffn_mode, val_prompts)

    return adapter, curve, {
        "config": config_name, "depth": depth, "attention_mode": attention_mode,
        "ffn_mode": ffn_mode, "epochs": epochs, "lr": lr, "seed": seed,
        "trainable_parameters": adapter.trainable_parameter_count(),
        "train_prompts": len(train_prompts), "val_prompts": len(val_prompts),
        "pre_train_loss_sample": pre_train_loss, "pre_val_loss": pre_val_loss,
        "post_train_loss_sample": post_train_loss, "post_val_loss": post_val_loss,
        "wall_clock_seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["shallow", "medium"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_prompts, val_prompts = load_split(seed=args.seed)
    print(f"Phase 0 retraining: {len(train_prompts)} train / {len(val_prompts)} val prompts "
          f"from {PROMPTS_PATH} (disjoint from the 60-prompt short_qa eval set)")

    model, tokenizer = load_model()

    summary = {}
    for config_name in args.configs:
        print(f"\n=== retraining '{config_name}' ({DEPTH_MAP[config_name]} layers, "
              f"{CONFIGURATIONS[config_name]['attention_mode']} attention, "
              f"{CONFIGURATIONS[config_name]['ffn_mode']} ffn) ===")

        adapter, curve, stats = finetune_config(
            config_name, model, tokenizer, train_prompts, val_prompts,
            epochs=args.epochs, lr=args.lr, seed=args.seed,
        )

        save_adapter(adapter, f"{OUTPUT_DIR}/{config_name}_adapter.pt")

        import pandas as pd
        pd.DataFrame(curve).to_csv(f"{OUTPUT_DIR}/{config_name}_training_curve.csv", index=False)

        summary[config_name] = stats
        print(f"  [{config_name}] pre val={stats['pre_val_loss']:.4f} -> "
              f"post val={stats['post_val_loss']:.4f} "
              f"({stats['trainable_parameters']:,} trainable params, "
              f"{stats['wall_clock_seconds']:.1f}s)")

    with open(f"{OUTPUT_DIR}/retraining_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nSaved adapters, training curves and summary to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
