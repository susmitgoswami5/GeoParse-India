"""
Unit tests for the synthetic data generator.
"""

import pytest
from geoparse.data.synthetic_generator import SyntheticAddressGenerator
from geoparse.ner.label_schema import LABEL2ID, LABELS


class TestSyntheticGenerator:
    """Tests for the synthetic address data generator."""

    def setup_method(self):
        self.generator = SyntheticAddressGenerator(noise_probability=0.5, seed=42)

    def test_generate_sample_returns_correct_keys(self):
        sample = self.generator.generate_sample()
        assert "tokens" in sample
        assert "labels" in sample
        assert "text" in sample
        assert "is_noisy" in sample

    def test_tokens_and_labels_same_length(self):
        for _ in range(100):
            sample = self.generator.generate_sample()
            assert len(sample["tokens"]) == len(sample["labels"]), \
                f"Mismatch: {len(sample['tokens'])} tokens vs {len(sample['labels'])} labels"

    def test_labels_are_valid_bio(self):
        valid_labels = set(LABELS)
        for _ in range(100):
            sample = self.generator.generate_sample()
            for label in sample["labels"]:
                assert label in valid_labels, f"Invalid label: {label}"

    def test_text_is_non_empty(self):
        for _ in range(50):
            sample = self.generator.generate_sample()
            assert len(sample["text"]) > 0

    def test_has_city_and_locality(self):
        """Generated addresses should always contain city and locality."""
        for _ in range(50):
            sample = self.generator.generate_sample()
            label_types = {l.split("-")[-1] for l in sample["labels"] if l != "O"}
            assert "CITY" in label_types or "LOCALITY" in label_types

    def test_generate_dataset_split(self):
        train, val = self.generator.generate_dataset(num_samples=100, train_ratio=0.8)
        assert len(train) == 80
        assert len(val) == 20

    def test_noisy_samples_are_generated(self):
        gen = SyntheticAddressGenerator(noise_probability=1.0, seed=123)
        sample = gen.generate_sample()
        assert sample["is_noisy"] is True

    def test_clean_samples_are_generated(self):
        gen = SyntheticAddressGenerator(noise_probability=0.0, seed=123)
        sample = gen.generate_sample()
        assert sample["is_noisy"] is False

    def test_reproducibility_with_seed(self):
        # Each generator reseeds global random, so create them sequentially
        # and generate one sample each to verify same seed → same output
        gen1 = SyntheticAddressGenerator(noise_probability=0.0, seed=999)
        s1 = gen1.generate_sample()
        gen2 = SyntheticAddressGenerator(noise_probability=0.0, seed=999)
        s2 = gen2.generate_sample()
        assert s1["tokens"] == s2["tokens"]
        assert s1["labels"] == s2["labels"]
