"""
Fine-tuning script for the address NER model using HuggingFace Trainer API.

Fine-tunes DistilBERT for token classification on synthetic Indian address data.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    from seqeval.metrics import (
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )
    HAS_SEQEVAL = True
except ImportError:
    HAS_SEQEVAL = False

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from geoparse.ner.dataset import AddressNERDataset
from geoparse.ner.label_schema import ID2LABEL, LABEL2ID, NUM_LABELS


def compute_metrics(eval_preds):
    """Compute seqeval metrics for NER evaluation."""
    if not HAS_SEQEVAL:
        return {}

    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=2)

    # Convert IDs back to labels, ignoring -100
    true_labels = []
    pred_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        true_seq = []
        pred_seq_labels = []
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue
            true_seq.append(ID2LABEL.get(label_id, "O"))
            pred_seq_labels.append(ID2LABEL.get(pred_id, "O"))
        true_labels.append(true_seq)
        pred_labels.append(pred_seq_labels)

    return {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels),
    }


def train_ner_model(
    train_data_path: str = "data/train.json",
    val_data_path: str = "data/val.json",
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "models/address_ner",
    num_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    max_length: int = 128,
    eval_steps: int = 500,
    save_steps: int = 500,
    warmup_ratio: float = 0.1,
    fp16: bool = False,
    seed: int = 42,
):
    """
    Fine-tune a transformer model for Indian address NER.

    Args:
        train_data_path: Path to training data JSON.
        val_data_path: Path to validation data JSON.
        model_name: HuggingFace model identifier.
        output_dir: Directory to save the trained model.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        weight_decay: Weight decay for AdamW.
        max_length: Maximum token sequence length.
        eval_steps: Evaluate every N steps.
        save_steps: Save checkpoint every N steps.
        warmup_ratio: Warmup ratio for learning rate scheduler.
        fp16: Whether to use mixed precision training.
        seed: Random seed.
    """
    print("=" * 60)
    print("  GeoParse-India: NER Model Training")
    print("=" * 60)

    # Load tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load datasets
    print(f"Loading training data from: {train_data_path}")
    train_dataset = AddressNERDataset.from_json(train_data_path, tokenizer, max_length)
    print(f"Loading validation data from: {val_data_path}")
    val_dataset = AddressNERDataset.from_json(val_data_path, tokenizer, max_length)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Load model
    print(f"Loading model: {model_name}")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        logging_steps=100,
        logging_dir=f"{output_dir}/logs",
        report_to="none",
        fp16=fp16,
        seed=seed,
        remove_unused_columns=False,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\n🚀 Starting training...")
    trainer.train()

    # Save best model
    best_model_path = f"{output_dir}/best"
    print(f"\n💾 Saving best model to: {best_model_path}")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)

    # Final evaluation
    print("\n📊 Final evaluation:")
    metrics = trainer.evaluate()
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # Save metrics
    metrics_path = Path(output_dir) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in metrics.items()}, f, indent=2)

    print(f"\n✅ Training complete! Model saved to {best_model_path}")
    return best_model_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train GeoParse-India NER model")
    parser.add_argument("--train-data", default="data/train.json")
    parser.add_argument("--val-data", default="data/val.json")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", default="models/address_ner")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    train_ner_model(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        fp16=args.fp16,
    )
