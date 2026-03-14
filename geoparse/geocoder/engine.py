"""
Geocoding engine — orchestrates the full pipeline.

NER → Phonetic Correction → Query Building → Spatial Lookup

Returns GeocodeResult with lat/lng, confidence, radius, and entity breakdown.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from geoparse.geocoder.query_builder import QueryBuilder
from geoparse.geocoder.spatial_index import SpatialIndex
from geoparse.phonetic.corrector import PhoneticCorrector


@dataclass
class GeocodeResult:
    """Result of geocoding an address."""
    lat: Optional[float] = None
    lng: Optional[float] = None
    confidence: float = 0.0
    radius_m: Optional[float] = None
    resolution: str = "none"
    needs_review: bool = False
    review_reason: Optional[str] = None
    h3_cell: Optional[str] = None

    # Extracted entities (after phonetic correction)
    entities: Dict[str, Any] = field(default_factory=dict)

    # Raw NER output (before correction)
    raw_entities: Dict[str, Any] = field(default_factory=dict)

    # Cleaned query used for geocoding
    query: Dict[str, Any] = field(default_factory=dict)

    # Original input text
    input_text: str = ""

    # Token-level predictions for visualization
    token_predictions: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "lat": self.lat,
            "lng": self.lng,
            "confidence": round(self.confidence, 4),
            "radius_m": self.radius_m,
            "resolution": self.resolution,
            "needs_review": self.needs_review,
            "review_reason": self.review_reason,
            "h3_cell": self.h3_cell,
            "entities": self.entities,
            "raw_entities": self.raw_entities,
            "query": self.query,
            "input_text": self.input_text,
            "token_predictions": self.token_predictions,
        }


class GeocodingEngine:
    """
    Full geocoding pipeline for Indian addresses.

    Orchestrates:
    1. NER entity extraction
    2. Phonetic correction of city/locality names
    3. Cleaned query construction
    4. Spatial lookup with H3 indexing
    5. Confidence scoring and manual review flagging
    """

    # Confidence threshold below which address is flagged for review
    REVIEW_CONFIDENCE_THRESHOLD = 0.5
    # Radius threshold (meters) above which address is flagged
    REVIEW_RADIUS_THRESHOLD = 2000

    def __init__(
        self,
        ner_parser: Optional[Any] = None,
        phonetic_corrector: Optional[PhoneticCorrector] = None,
        query_builder: Optional[QueryBuilder] = None,
        spatial_index: Optional[SpatialIndex] = None,
    ):
        """
        Args:
            ner_parser: AddressNERParser instance. If None, uses mock NER.
            phonetic_corrector: PhoneticCorrector instance.
            query_builder: QueryBuilder instance.
            spatial_index: SpatialIndex instance.
        """
        self.ner_parser = ner_parser
        self.phonetic_corrector = phonetic_corrector or PhoneticCorrector()
        self.query_builder = query_builder or QueryBuilder()
        self.spatial_index = spatial_index or SpatialIndex()

    def geocode(self, address_text: str) -> GeocodeResult:
        """
        Full geocoding pipeline for a single address.

        Args:
            address_text: Raw unstructured Indian address text.

        Returns:
            GeocodeResult with location, confidence, and entity breakdown.
        """
        result = GeocodeResult(input_text=address_text)

        # Step 1: NER entity extraction
        if self.ner_parser:
            ner_output = self.ner_parser.parse(address_text)
            raw_entities = ner_output["entities"]
            result.token_predictions = ner_output.get("token_predictions", [])
        else:
            # Mock NER: use rule-based extraction as fallback
            raw_entities = self._rule_based_extract(address_text)
            result.token_predictions = []

        result.raw_entities = {
            k: {"text": v["text"], "confidence": v["confidence"]}
            for k, v in raw_entities.items()
        }

        # Step 2: Phonetic correction
        corrected_entities = self.phonetic_corrector.correct_entities(raw_entities)
        result.entities = corrected_entities

        # Step 3: Build cleaned geocoding query
        query_input = {
            k: {"text": v["text"], "confidence": v.get("confidence", 0.5)}
            for k, v in corrected_entities.items()
        }
        query_result = self.query_builder.build_query(query_input)
        result.query = query_result

        # Step 4: Spatial lookup
        city = corrected_entities.get("CITY", {}).get("text")
        locality = corrected_entities.get("LOCALITY", {}).get("text")
        pincode = corrected_entities.get("PINCODE", {}).get("text")

        geo_result = self.spatial_index.geocode(
            city=city, locality=locality, pincode=pincode
        )

        result.lat = geo_result["lat"]
        result.lng = geo_result["lng"]
        result.radius_m = geo_result["radius_m"]
        result.confidence = geo_result["confidence"]
        result.resolution = geo_result["resolution"]
        result.h3_cell = geo_result.get("h3_cell")

        # Step 5: Flag for manual review if needed
        if result.confidence < self.REVIEW_CONFIDENCE_THRESHOLD:
            result.needs_review = True
            result.review_reason = f"Low confidence ({result.confidence:.2f})"
        elif result.radius_m and result.radius_m > self.REVIEW_RADIUS_THRESHOLD:
            result.needs_review = True
            result.review_reason = f"Large radius ({result.radius_m}m) — provide better landmark"
        elif not query_result["is_valid"]:
            result.needs_review = True
            result.review_reason = "Insufficient address components extracted"

        return result

    def _rule_based_extract(self, text: str) -> Dict[str, Dict]:
        """
        Fallback rule-based entity extraction.

        Used when no NER model is loaded. Applies simple heuristics
        to extract entities from the address text.
        """
        import re

        entities = {}
        text_lower = text.lower()

        # Extract pincode (6-digit number)
        pincode_match = re.search(r'\b(\d{6})\b', text)
        if pincode_match:
            entities["PINCODE"] = {
                "text": pincode_match.group(1),
                "confidence": 0.95,
            }

        # Match known cities
        from geoparse.data.address_components import CITIES, TRANSLITERATIONS
        for city_data in CITIES:
            city_name = city_data["name"]
            if city_name.lower() in text_lower:
                entities["CITY"] = {"text": city_name, "confidence": 0.9}
                break
            # Check variants
            if city_name in TRANSLITERATIONS:
                for variant in TRANSLITERATIONS[city_name]:
                    if variant.lower() in text_lower:
                        entities["CITY"] = {"text": variant, "confidence": 0.7}
                        break
                if "CITY" in entities:
                    break

        # Match known localities
        from geoparse.data.address_components import LOCALITIES
        for city_name, locs in LOCALITIES.items():
            for loc in locs:
                if loc.lower() in text_lower:
                    entities["LOCALITY"] = {"text": loc, "confidence": 0.8}
                    break
            if "LOCALITY" in entities:
                break

        # Match house numbers
        house_match = re.search(
            r'\b(?:flat|house|plot|h\.?no\.?|d\.?no\.?|#)\s*\.?\s*(\d[\d\-/a-zA-Z]*)\b',
            text, re.IGNORECASE
        )
        if house_match:
            entities["HOUSE_NO"] = {
                "text": house_match.group(0).strip(),
                "confidence": 0.85,
            }

        # Match landmarks (near/opp/behind + phrase)
        landmark_match = re.search(
            r'(?:near|opp(?:osite)?|behind|beside|next\s+to|in\s+front\s+of)\s+(.+?)(?:,|$)',
            text, re.IGNORECASE
        )
        if landmark_match:
            entities["LANDMARK"] = {
                "text": landmark_match.group(1).strip(),
                "confidence": 0.7,
            }

        # Match streets
        street_match = re.search(
            r'(\d+(?:st|nd|rd|th)?\s+(?:cross|main|street|road|lane|marg|gali))',
            text, re.IGNORECASE
        )
        if not street_match:
            street_match = re.search(
                r'((?:MG|GT|NH|SH)\s*(?:road|rd|highway))',
                text, re.IGNORECASE
            )
        if street_match:
            entities["STREET"] = {
                "text": street_match.group(1).strip(),
                "confidence": 0.75,
            }

        return entities

    def geocode_batch(self, addresses: List[str]) -> List[GeocodeResult]:
        """Geocode a batch of addresses."""
        return [self.geocode(addr) for addr in addresses]
