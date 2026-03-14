"""
Phonetic correction layer for Indian address transliterations.

Uses Double Metaphone + Jaro-Winkler similarity to fuzzy-match misspelled
Indian city, locality, and landmark names against a canonical dictionary.
Handles common variations like 'Bnglr' -> 'Bengaluru', 'Dlhi' -> 'Delhi'.
"""

from typing import Dict, List, Optional, Tuple

try:
    from metaphone import doublemetaphone
    HAS_METAPHONE = True
except ImportError:
    HAS_METAPHONE = False

try:
    import jellyfish
    HAS_JELLYFISH = True
except ImportError:
    HAS_JELLYFISH = False

from geoparse.data.address_components import CITIES, LOCALITIES, DEFAULT_LOCALITIES, TRANSLITERATIONS


class PhoneticCorrector:
    """
    Corrects phonetic misspellings in Indian address entities.

    Maintains a canonical dictionary of city and locality names and uses
    Double Metaphone encoding + Jaro-Winkler similarity for fuzzy matching.
    """

    def __init__(self, similarity_threshold: float = 0.80):
        """
        Args:
            similarity_threshold: Minimum Jaro-Winkler similarity for a match (0.0-1.0).
        """
        self.similarity_threshold = similarity_threshold

        # Build canonical dictionaries
        self.canonical_cities: Dict[str, Dict] = {}
        self.city_variants: Dict[str, str] = {}  # variant -> canonical
        self.canonical_localities: Dict[str, str] = {}  # locality -> city

        self._build_dictionaries()

    def _build_dictionaries(self):
        """Build canonical name dictionaries from address components."""
        # Cities
        for city_data in CITIES:
            name = city_data["name"]
            self.canonical_cities[name.lower()] = city_data

            # Add known transliteration variants
            if name in TRANSLITERATIONS:
                for variant in TRANSLITERATIONS[name]:
                    self.city_variants[variant.lower()] = name

            # Also add itself
            self.city_variants[name.lower()] = name

        # Localities
        for city_name, locs in LOCALITIES.items():
            for loc in locs:
                self.canonical_localities[loc.lower()] = city_name

        for loc in DEFAULT_LOCALITIES:
            if loc.lower() not in self.canonical_localities:
                self.canonical_localities[loc.lower()] = "Unknown"

        # Build metaphone index for fuzzy matching
        self.city_metaphones: Dict[str, List[str]] = {}
        if HAS_METAPHONE:
            for name in self.canonical_cities:
                codes = doublemetaphone(name)
                for code in codes:
                    if code:
                        if code not in self.city_metaphones:
                            self.city_metaphones[code] = []
                        self.city_metaphones[code].append(name)

    def correct_city(self, text: str) -> Tuple[str, float]:
        """
        Correct a city name using phonetic matching.

        Args:
            text: Potentially misspelled city name.

        Returns:
            Tuple of (corrected_name, confidence_score).
        """
        text_lower = text.lower().strip()

        # 1. Exact match
        if text_lower in self.canonical_cities:
            return self.canonical_cities[text_lower]["name"], 1.0

        # 2. Known variant match
        if text_lower in self.city_variants:
            canonical = self.city_variants[text_lower]
            return canonical, 0.95

        # 3. Metaphone-based fuzzy match
        best_match = None
        best_score = 0.0

        if HAS_METAPHONE and HAS_JELLYFISH:
            codes = doublemetaphone(text_lower)
            candidates = set()
            for code in codes:
                if code and code in self.city_metaphones:
                    candidates.update(self.city_metaphones[code])

            for candidate in candidates:
                score = jellyfish.jaro_winkler_similarity(text_lower, candidate)
                if score > best_score:
                    best_score = score
                    best_match = candidate

        # 4. Brute-force Jaro-Winkler if no metaphone match
        if (best_match is None or best_score < self.similarity_threshold) and HAS_JELLYFISH:
            for canonical_name in self.canonical_cities:
                score = jellyfish.jaro_winkler_similarity(text_lower, canonical_name)
                if score > best_score:
                    best_score = score
                    best_match = canonical_name

        if best_match and best_score >= self.similarity_threshold:
            return self.canonical_cities[best_match]["name"], round(best_score, 4)

        # No match found — return original
        return text, 0.0

    def correct_locality(self, text: str) -> Tuple[str, float]:
        """
        Correct a locality name using fuzzy matching.

        Args:
            text: Potentially misspelled locality name.

        Returns:
            Tuple of (corrected_name, confidence_score).
        """
        text_lower = text.lower().strip()

        # Exact match
        if text_lower in self.canonical_localities:
            # Return with original casing from data
            for city_name, locs in LOCALITIES.items():
                for loc in locs:
                    if loc.lower() == text_lower:
                        return loc, 1.0
            for loc in DEFAULT_LOCALITIES:
                if loc.lower() == text_lower:
                    return loc, 1.0
            return text, 0.9

        # Fuzzy match
        best_match = None
        best_score = 0.0

        if HAS_JELLYFISH:
            all_localities = list(self.canonical_localities.keys())
            for loc_name in all_localities:
                score = jellyfish.jaro_winkler_similarity(text_lower, loc_name)
                if score > best_score:
                    best_score = score
                    best_match = loc_name

        if best_match and best_score >= self.similarity_threshold:
            # Get original casing
            for city_name, locs in LOCALITIES.items():
                for loc in locs:
                    if loc.lower() == best_match:
                        return loc, round(best_score, 4)
            for loc in DEFAULT_LOCALITIES:
                if loc.lower() == best_match:
                    return loc, round(best_score, 4)
            return best_match.title(), round(best_score, 4)

        return text, 0.0

    def correct_entities(self, entities: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Apply phonetic corrections to all extracted entities.

        Args:
            entities: Dict of entity_type -> {text, confidence}

        Returns:
            Corrected entities dict with original values preserved.
        """
        corrected = {}

        for entity_type, entity_data in entities.items():
            text = entity_data["text"]
            confidence = entity_data["confidence"]

            if entity_type == "CITY":
                corrected_text, correction_conf = self.correct_city(text)
                corrected[entity_type] = {
                    "text": corrected_text,
                    "original": text if corrected_text != text else None,
                    "confidence": confidence,
                    "correction_confidence": correction_conf,
                    "was_corrected": corrected_text != text,
                }
            elif entity_type == "LOCALITY":
                corrected_text, correction_conf = self.correct_locality(text)
                corrected[entity_type] = {
                    "text": corrected_text,
                    "original": text if corrected_text != text else None,
                    "confidence": confidence,
                    "correction_confidence": correction_conf,
                    "was_corrected": corrected_text != text,
                }
            else:
                corrected[entity_type] = {
                    "text": text,
                    "original": None,
                    "confidence": confidence,
                    "correction_confidence": 1.0,
                    "was_corrected": False,
                }

        return corrected
