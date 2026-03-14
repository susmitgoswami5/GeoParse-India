"""
Unit tests for the geocoding query builder.
"""

import pytest
from geoparse.geocoder.query_builder import QueryBuilder


class TestQueryBuilder:
    """Tests for the cleaned query builder."""

    def setup_method(self):
        self.builder = QueryBuilder()

    def test_full_address_query(self):
        entities = {
            "HOUSE_NO": {"text": "42/B", "confidence": 0.9},
            "STREET": {"text": "MG Road", "confidence": 0.85},
            "LANDMARK": {"text": "SBI Bank", "confidence": 0.8},
            "LOCALITY": {"text": "Koramangala", "confidence": 0.9},
            "CITY": {"text": "Bengaluru", "confidence": 0.95},
            "PINCODE": {"text": "560034", "confidence": 0.9},
        }
        result = self.builder.build_query(entities)
        assert result["is_valid"] is True
        assert result["num_entities"] >= 4
        assert "Bengaluru" in result["query_string"]
        assert "Koramangala" in result["query_string"]

    def test_minimal_valid_query(self):
        entities = {
            "LOCALITY": {"text": "Whitefield", "confidence": 0.8},
            "CITY": {"text": "Bengaluru", "confidence": 0.9},
        }
        result = self.builder.build_query(entities)
        assert result["is_valid"] is True

    def test_invalid_query_single_entity(self):
        entities = {
            "LANDMARK": {"text": "SBI Bank", "confidence": 0.8},
        }
        result = self.builder.build_query(entities)
        assert result["is_valid"] is False

    def test_low_confidence_entities_filtered(self):
        entities = {
            "CITY": {"text": "Bengaluru", "confidence": 0.1},  # Below threshold
            "LOCALITY": {"text": "Unknown", "confidence": 0.1},
        }
        result = self.builder.build_query(entities)
        assert result["num_entities"] == 0
        assert len(result["missing_components"]) > 0

    def test_query_string_order(self):
        entities = {
            "CITY": {"text": "Delhi", "confidence": 0.9},
            "HOUSE_NO": {"text": "#18", "confidence": 0.9},
            "LOCALITY": {"text": "Connaught Place", "confidence": 0.9},
        }
        result = self.builder.build_query(entities)
        query = result["query_string"]
        # House number should come before city in geocoding order
        house_pos = query.find("#18")
        city_pos = query.find("Delhi")
        assert house_pos < city_pos

    def test_pincode_only_is_valid(self):
        entities = {
            "PINCODE": {"text": "560066", "confidence": 0.95},
            "LANDMARK": {"text": "Big Tree", "confidence": 0.7},
        }
        result = self.builder.build_query(entities)
        # Pincode counts as an area entity
        assert result["is_valid"] is True

    def test_confidence_is_average(self):
        entities = {
            "CITY": {"text": "Mumbai", "confidence": 0.8},
            "LOCALITY": {"text": "Andheri", "confidence": 0.6},
        }
        result = self.builder.build_query(entities)
        assert 0.6 <= result["confidence"] <= 0.8
