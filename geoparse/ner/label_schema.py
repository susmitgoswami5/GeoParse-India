"""
BIO label schema for Indian address NER.

Defines the complete set of entity labels used for token classification,
along with utility functions for label ↔ ID mapping.
"""

from typing import Dict, List

# Complete BIO label set for Indian address parsing
LABELS: List[str] = [
    "O",            # Outside any entity
    "B-HOUSE_NO",   # Beginning of house/flat number
    "I-HOUSE_NO",   # Inside house/flat number
    "B-BUILDING",   # Beginning of building/apartment name
    "I-BUILDING",   # Inside building/apartment name
    "B-STREET",     # Beginning of street/road name
    "I-STREET",     # Inside street/road name
    "B-LANDMARK",   # Beginning of landmark
    "I-LANDMARK",   # Inside landmark
    "B-LOCALITY",   # Beginning of locality/sub-locality
    "I-LOCALITY",   # Inside locality/sub-locality
    "B-CITY",       # Beginning of city name
    "I-CITY",       # Inside city name
    "B-STATE",      # Beginning of state name
    "I-STATE",      # Inside state name
    "B-PINCODE",    # Beginning of pincode
    "I-PINCODE",    # Inside pincode
]

# Label to ID mapping
LABEL2ID: Dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}

# ID to label mapping
ID2LABEL: Dict[int, str] = {idx: label for idx, label in enumerate(LABELS)}

# Number of labels
NUM_LABELS: int = len(LABELS)

# Entity types (without BIO prefix)
ENTITY_TYPES: List[str] = [
    "HOUSE_NO", "BUILDING", "STREET", "LANDMARK",
    "LOCALITY", "CITY", "STATE", "PINCODE",
]


def get_entity_type(label: str) -> str:
    """Extract entity type from BIO label (e.g., 'B-CITY' -> 'CITY')."""
    if label == "O":
        return "O"
    return label.split("-", 1)[1]


def is_begin_label(label: str) -> bool:
    """Check if label is a Begin tag."""
    return label.startswith("B-")


def is_inside_label(label: str) -> bool:
    """Check if label is an Inside tag."""
    return label.startswith("I-")
