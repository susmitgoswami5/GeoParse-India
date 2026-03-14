"""
Unit tests for the phonetic correction layer.
"""

import pytest
from geoparse.phonetic.corrector import PhoneticCorrector


class TestPhoneticCorrector:
    """Tests for the phonetic corrector."""

    def setup_method(self):
        self.corrector = PhoneticCorrector(similarity_threshold=0.80)

    # ---- City Corrections ----

    def test_exact_city_match(self):
        name, conf = self.corrector.correct_city("Mumbai")
        assert name == "Mumbai"
        assert conf == 1.0

    def test_case_insensitive_city(self):
        name, conf = self.corrector.correct_city("mumbai")
        assert name == "Mumbai"
        assert conf == 1.0

    def test_known_variant_bangalore(self):
        name, conf = self.corrector.correct_city("Bangalore")
        assert name == "Bengaluru"
        assert conf == 0.95

    def test_known_variant_bnglr(self):
        name, conf = self.corrector.correct_city("Bnglr")
        assert name == "Bengaluru"
        assert conf == 0.95

    def test_known_variant_calcutta(self):
        name, conf = self.corrector.correct_city("Calcutta")
        assert name == "Kolkata"
        assert conf == 0.95

    def test_known_variant_bombay(self):
        name, conf = self.corrector.correct_city("Bombay")
        assert name == "Mumbai"
        assert conf == 0.95

    def test_known_variant_madras(self):
        name, conf = self.corrector.correct_city("Madras")
        assert name == "Chennai"
        assert conf == 0.95

    def test_known_variant_trivandrum(self):
        name, conf = self.corrector.correct_city("Trivandrum")
        assert name == "Thiruvananthapuram"
        assert conf == 0.95

    def test_known_variant_gurgaon(self):
        name, conf = self.corrector.correct_city("Gurgaon")
        assert name == "Gurugram"
        assert conf == 0.95

    def test_known_variant_poona(self):
        name, conf = self.corrector.correct_city("Poona")
        assert name == "Pune"
        assert conf == 0.95

    def test_known_variant_baroda(self):
        name, conf = self.corrector.correct_city("Baroda")
        assert name == "Vadodara"
        assert conf == 0.95

    def test_unknown_city_returned_as_is(self):
        name, conf = self.corrector.correct_city("XyzNonexistentCity")
        assert conf < 0.8  # Should not match

    # ---- Locality Corrections ----

    def test_exact_locality_match(self):
        name, conf = self.corrector.correct_locality("Koramangala")
        assert name == "Koramangala"
        assert conf == 1.0

    def test_case_insensitive_locality(self):
        name, conf = self.corrector.correct_locality("koramangala")
        assert name == "Koramangala"
        assert conf == 1.0

    # ---- Entity Correction ----

    def test_correct_entities_city(self):
        entities = {
            "CITY": {"text": "Bangalore", "confidence": 0.9},
            "LOCALITY": {"text": "Whitefield", "confidence": 0.85},
        }
        corrected = self.corrector.correct_entities(entities)
        assert corrected["CITY"]["text"] == "Bengaluru"
        assert corrected["CITY"]["was_corrected"] is True
        assert corrected["CITY"]["original"] == "Bangalore"

    def test_correct_entities_no_correction_needed(self):
        entities = {
            "CITY": {"text": "Mumbai", "confidence": 0.9},
        }
        corrected = self.corrector.correct_entities(entities)
        assert corrected["CITY"]["text"] == "Mumbai"
        assert corrected["CITY"]["was_corrected"] is False

    def test_correct_entities_preserves_other_types(self):
        entities = {
            "HOUSE_NO": {"text": "42/B", "confidence": 0.95},
            "LANDMARK": {"text": "SBI Bank", "confidence": 0.8},
        }
        corrected = self.corrector.correct_entities(entities)
        assert corrected["HOUSE_NO"]["text"] == "42/B"
        assert corrected["LANDMARK"]["text"] == "SBI Bank"
