"""
PyTorch Dataset for Indian address NER token classification.

Handles sub-word tokenization label propagation — assigns the original
label to the first sub-token and -100 to subsequent sub-tokens.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from geoparse.ner.label_schema import LABEL2ID


class AddressNERDataset(Dataset):
    """
    PyTorch Dataset for address NER training.

    Tokenizes word-level tokens with a HuggingFace tokenizer and aligns
    BIO labels to sub-word tokens. The first sub-token of each word gets
    the original label, and all subsequent sub-tokens get -100 (ignored
    in cross-entropy loss).
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: Any,
        max_length: int = 128,
    ):
        """
        Args:
            data: List of dicts with 'tokens' and 'labels' keys.
            tokenizer: HuggingFace tokenizer instance.
            max_length: Maximum sequence length for tokenization.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        tokens = sample["tokens"]
        labels = sample["labels"]

        # Tokenize with word-level alignment
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Align labels to sub-tokens
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # Special token ([CLS], [SEP], [PAD])
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                # First sub-token of a new word: use the original label
                if word_idx < len(labels):
                    aligned_labels.append(LABEL2ID.get(labels[word_idx], 0))
                else:
                    aligned_labels.append(-100)
            else:
                # Subsequent sub-token of the same word: ignore in loss
                aligned_labels.append(-100)
            previous_word_idx = word_idx

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }

    @classmethod
    def from_json(
        cls,
        path: str,
        tokenizer: Any,
        max_length: int = 128,
    ) -> "AddressNERDataset":
        """Load dataset from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(data=data, tokenizer=tokenizer, max_length=max_length)
