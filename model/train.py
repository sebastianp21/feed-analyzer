"""Fine-tune one model on data/processed/{train,val}.csv.

Usage (from the model/ folder):
    python train.py --model modernbert
    python train.py --model roberta --epochs 4
    python train.py --model distilbert --out /content/drive/MyDrive/feed_analyzer/distilbert
    python train.py --model distilbert --epochs 1 --limit 200   # quick CPU check

The saved model has our 4 categories as its labels, so the server detects it
as a fine-tuned classifier just by pointing MODEL_PATH at the output folder.
The test split is NOT used here; it is only for evaluate.py (Phase 3).
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from prepare_data import CATEGORIES, PROCESSED_DIR, SEED

# Short name -> (Hugging Face model id, learning rate)
MODELS = {
    "modernbert": ("answerdotai/ModernBERT-base", 5e-5),
    "roberta": ("cardiffnlp/twitter-roberta-base-2022-154m", 2e-5),
    "distilbert": ("distilbert/distilbert-base-uncased", 5e-5),
}
MAX_LENGTH = 128  # tokens; almost all tweets are shorter

LABEL2ID = {name: i for i, name in enumerate(CATEGORIES)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", help="output folder (default: output/<model>)")
    parser.add_argument("--limit", type=int,
                        help="use only this many train/val posts (quick test run)")
    args = parser.parse_args()

    model_id, learning_rate = MODELS[args.model]
    out_dir = Path(args.out or f"output/{args.model}")

    # Load the CSVs made by prepare_data.py and turn category names into ids.
    data = load_dataset(
        "csv",
        data_files={
            "train": str(PROCESSED_DIR / "train.csv"),
            "val": str(PROCESSED_DIR / "val.csv"),
        },
    )
    if args.limit:
        for split in data:
            data[split] = data[split].shuffle(seed=SEED).select(range(args.limit))
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def tokenize(batch):
        encoded = tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)
        encoded["labels"] = [LABEL2ID[c] for c in batch["category"]]
        return encoded

    data = data.map(tokenize, batched=True, remove_columns=data["train"].column_names)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(CATEGORIES), id2label=ID2LABEL, label2id=LABEL2ID
    )

    # Warm up the learning rate over the first 10% of steps. (Computed by hand
    # because the warmup_ratio option was removed in transformers 5.)
    total_steps = math.ceil(len(data["train"]) / args.batch_size) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,  # keep the epoch with the best val macro-F1
        metric_for_best_model="macro_f1",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),  # faster on Colab's T4
        seed=SEED,
        logging_steps=50,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Save the best model + tokenizer, plus its validation scores.
    trainer.save_model(str(out_dir))
    shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)  # big, no longer needed
    val_metrics = trainer.evaluate()
    (out_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2))
    print(f"Saved {args.model} to {out_dir}")
    print(f"Validation: accuracy={val_metrics['eval_accuracy']:.3f} "
          f"macro-F1={val_metrics['eval_macro_f1']:.3f}")


if __name__ == "__main__":
    main()
