"""
Unit tests for the NER label schema.
"""

import pytest
from geoparse.ner.label_schema import (
    LABELS, LABEL2ID, ID2LABEL, NUM_LABELS,
    ENTITY_TYPES, get_entity_type, is_begin_label, is_inside_label,
)


class TestLabelSchema:
    """Tests for the BIO label schema."""

    def test_label_count(self):
        assert NUM_LABELS == 17

    def test_label2id_roundtrip(self):
        for label in LABELS:
            label_id = LABEL2ID[label]
            assert ID2LABEL[label_id] == label

    def test_id2label_roundtrip(self):
        for idx in range(NUM_LABELS):
            label = ID2LABEL[idx]
            assert LABEL2ID[label] == idx

    def test_o_label_is_zero(self):
        assert LABEL2ID["O"] == 0

    def test_entity_types(self):
        expected = {"HOUSE_NO", "BUILDING", "STREET", "LANDMARK",
                    "LOCALITY", "CITY", "STATE", "PINCODE"}
        assert set(ENTITY_TYPES) == expected

    def test_get_entity_type(self):
        assert get_entity_type("B-CITY") == "CITY"
        assert get_entity_type("I-STREET") == "STREET"
        assert get_entity_type("O") == "O"

    def test_is_begin_label(self):
        assert is_begin_label("B-CITY") is True
        assert is_begin_label("I-CITY") is False
        assert is_begin_label("O") is False

    def test_is_inside_label(self):
        assert is_inside_label("I-STREET") is True
        assert is_inside_label("B-STREET") is False
        assert is_inside_label("O") is False

    def test_all_entity_types_have_b_and_i(self):
        for etype in ENTITY_TYPES:
            assert f"B-{etype}" in LABELS
            assert f"I-{etype}" in LABELS
