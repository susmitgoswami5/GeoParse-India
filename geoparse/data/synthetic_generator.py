"""
Synthetic Indian address generator for NER training data.

Generates structured addresses from component dictionaries, applies noise
functions (random deletion, reordering, transliteration swap, case corruption,
punctuation stripping), and outputs BIO-tagged token sequences.
"""

import json
import random
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from geoparse.data.address_components import (
    BUILDING_NAMES,
    CITIES,
    DIRECTION_MARKERS,
    LANDMARKS,
    STREET_NAMES,
    TRANSLITERATIONS,
    generate_house_number,
    get_localities,
    get_noisy_variant,
)


class SyntheticAddressGenerator:
    """
    Generates synthetic Indian addresses with BIO NER labels.

    The generator creates realistic address strings by composing hierarchical
    components (house number, building, street, landmark, locality, city, state,
    pincode) and then optionally applying noise transformations to simulate
    real-world "dirty" address data.
    """

    def __init__(self, noise_probability: float = 0.6, seed: Optional[int] = None):
        """
        Args:
            noise_probability: Probability of applying noise to each address (0.0 - 1.0).
            seed: Random seed for reproducibility.
        """
        self.noise_probability = noise_probability
        if seed is not None:
            random.seed(seed)

        self.cities = CITIES
        self.landmarks = LANDMARKS
        self.street_names = STREET_NAMES
        self.building_names = BUILDING_NAMES
        self.direction_markers = DIRECTION_MARKERS

    def _generate_clean_address(self) -> Tuple[List[str], List[str]]:
        """
        Generate a single clean structured address with BIO labels.

        Returns:
            Tuple of (tokens, labels) where each token has a corresponding BIO label.
        """
        city_data = random.choice(self.cities)
        city_name = city_data["name"]
        state = city_data["state"]
        pincode = random.choice(city_data["pincodes"])
        localities = get_localities(city_name)
        locality = random.choice(localities)

        tokens: List[str] = []
        labels: List[str] = []

        # Build address components - randomly include/exclude some
        components: List[Tuple[str, str]] = []

        # 1. House Number (70% chance)
        if random.random() < 0.7:
            house_no = generate_house_number()
            components.append((house_no, "HOUSE_NO"))

        # 2. Building Name (40% chance)
        if random.random() < 0.4:
            building = random.choice(self.building_names)
            components.append((building, "BUILDING"))

        # 3. Street (60% chance)
        if random.random() < 0.6:
            street = random.choice(self.street_names)
            components.append((street, "STREET"))

        # 4. Landmark with direction marker (50% chance)
        if random.random() < 0.5:
            landmark = random.choice(self.landmarks)
            if random.random() < 0.6:
                marker = random.choice(self.direction_markers)
                landmark = f"{marker} {landmark}"
            components.append((landmark, "LANDMARK"))

        # 5. Locality (90% chance — very common)
        if random.random() < 0.9:
            components.append((locality, "LOCALITY"))

        # 6. City (95% chance)
        if random.random() < 0.95:
            components.append((city_name, "CITY"))

        # 7. State (30% chance)
        if random.random() < 0.3:
            components.append((state, "STATE"))

        # 8. Pincode (60% chance)
        if random.random() < 0.6:
            components.append((pincode, "PINCODE"))

        # Ensure at least locality + city
        entity_types_present = {c[1] for c in components}
        if "LOCALITY" not in entity_types_present:
            components.append((locality, "LOCALITY"))
        if "CITY" not in entity_types_present:
            components.append((city_name, "CITY"))

        # Convert components to tokens with BIO labels
        for component_text, entity_type in components:
            words = component_text.split()
            for i, word in enumerate(words):
                tokens.append(word)
                if entity_type == "LANDMARK" and word in self.direction_markers:
                    labels.append("O")  # Direction markers are not part of the entity
                elif i == 0:
                    labels.append(f"B-{entity_type}")
                else:
                    labels.append(f"I-{entity_type}")

            # Add separator (comma or space) between components
            if random.random() < 0.5:
                tokens.append(",")
                labels.append("O")

        # Remove trailing comma if present
        if tokens and tokens[-1] == ",":
            tokens.pop()
            labels.pop()

        return tokens, labels

    def _apply_noise(self, tokens: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
        """
        Apply realistic noise transformations to the address.

        Noise transformations:
        - Transliteration swaps (e.g., "Road" -> "Rd")
        - Case corruption (random upper/lower)
        - Punctuation stripping
        - Random comma removal
        - Token concatenation (removing spaces)
        """
        noisy_tokens = []
        noisy_labels = []

        for token, label in zip(tokens, labels):
            # Skip commas sometimes
            if token == "," and random.random() < 0.4:
                continue

            # Transliteration swap (30% chance per token)
            if random.random() < 0.3:
                token = get_noisy_variant(token)

            # Case corruption (20% chance)
            if random.random() < 0.2:
                case_fn = random.choice([str.lower, str.upper, str.title])
                token = case_fn(token)

            # Strip periods/dots (15% chance)
            if random.random() < 0.15:
                token = token.replace(".", "")

            # Add typo (5% chance) — swap two adjacent characters
            if len(token) > 3 and random.random() < 0.05:
                pos = random.randint(0, len(token) - 2)
                token = token[:pos] + token[pos + 1] + token[pos] + token[pos + 2:]

            noisy_tokens.append(token)
            noisy_labels.append(label)

        return noisy_tokens, noisy_labels

    def _reorder_components(self, tokens: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
        """
        Randomly reorder address components (simulating non-standard ordering).
        Groups tokens by entity spans and shuffles them.
        """
        # Group into entity spans
        spans: List[Tuple[List[str], List[str]]] = []
        current_tokens: List[str] = []
        current_labels: List[str] = []

        for token, label in zip(tokens, labels):
            if label.startswith("B-") and current_tokens:
                spans.append((current_tokens, current_labels))
                current_tokens = [token]
                current_labels = [label]
            else:
                current_tokens.append(token)
                current_labels.append(label)

        if current_tokens:
            spans.append((current_tokens, current_labels))

        # Shuffle spans (30% chance)
        if random.random() < 0.3:
            random.shuffle(spans)

        # Flatten back
        result_tokens: List[str] = []
        result_labels: List[str] = []
        for span_tokens, span_labels in spans:
            result_tokens.extend(span_tokens)
            result_labels.extend(span_labels)

        return result_tokens, result_labels

    def generate_sample(self) -> Dict[str, Any]:
        """
        Generate a single training sample.

        Returns:
            Dict with keys: 'tokens', 'labels', 'text', 'is_noisy'
        """
        tokens, labels = self._generate_clean_address()

        is_noisy = random.random() < self.noise_probability
        if is_noisy:
            tokens, labels = self._reorder_components(tokens, labels)
            tokens, labels = self._apply_noise(tokens, labels)

        # Build text from tokens
        text_parts = []
        for t in tokens:
            if t == ",":
                if text_parts:
                    text_parts[-1] = text_parts[-1] + ","
            else:
                text_parts.append(t)

        text = " ".join(text_parts)

        # Rebuild clean token list (remove standalone commas, merge)
        clean_tokens = []
        clean_labels = []
        for t, l in zip(tokens, labels):
            if t == ",":
                if clean_tokens:
                    clean_tokens[-1] = clean_tokens[-1] + ","
                continue
            clean_tokens.append(t)
            clean_labels.append(l)

        return {
            "tokens": clean_tokens,
            "labels": clean_labels,
            "text": text,
            "is_noisy": is_noisy,
        }

    def generate_dataset(
        self, num_samples: int = 50000, train_ratio: float = 0.85
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate a full train/validation dataset.

        Args:
            num_samples: Total number of samples.
            train_ratio: Fraction used for training.

        Returns:
            Tuple of (train_samples, val_samples)
        """
        samples = [self.generate_sample() for _ in range(num_samples)]
        random.shuffle(samples)

        split_idx = int(len(samples) * train_ratio)
        return samples[:split_idx], samples[split_idx:]

    def save_dataset(
        self,
        output_dir: str,
        num_samples: int = 50000,
        train_ratio: float = 0.85,
    ) -> Tuple[str, str]:
        """
        Generate and save dataset to JSON files.

        Returns:
            Tuple of (train_path, val_path)
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        train_data, val_data = self.generate_dataset(num_samples, train_ratio)

        train_path = out / "train.json"
        val_path = out / "val.json"

        with open(train_path, "w") as f:
            json.dump(train_data, f, indent=2)

        with open(val_path, "w") as f:
            json.dump(val_data, f, indent=2)

        print(f"Generated {len(train_data)} training + {len(val_data)} validation samples")
        print(f"Saved to: {train_path}, {val_path}")

        return str(train_path), str(val_path)
