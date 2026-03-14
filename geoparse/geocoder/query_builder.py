"""
Cleaned Query builder for geocoding from NER outputs.

Constructs the optimal geocoding query from highest-confidence entities
using a priority hierarchy: Pincode > City > Locality > Landmark > Street.
"""

from typing import Any, Dict, List, Optional


# Priority order for constructing geocoding queries
ENTITY_PRIORITY = [
    "PINCODE",    # Most specific
    "CITY",       # Required for disambiguation
    "LOCALITY",   # Sub-city area
    "LANDMARK",   # Point of interest
    "STREET",     # Street name
    "BUILDING",   # Building name
    "HOUSE_NO",   # Most granular
]

# Minimum confidence threshold per entity type
CONFIDENCE_THRESHOLDS = {
    "PINCODE": 0.5,
    "CITY": 0.4,
    "LOCALITY": 0.4,
    "LANDMARK": 0.5,
    "STREET": 0.5,
    "BUILDING": 0.5,
    "HOUSE_NO": 0.5,
    "STATE": 0.4,
}


class QueryBuilder:
    """
    Builds cleaned geocoding queries from NER-extracted entities.

    Instead of sending the whole messy address string to a geocoder,
    constructs a clean, structured query using only high-confidence entities.
    """

    def __init__(
        self,
        confidence_thresholds: Optional[Dict[str, float]] = None,
        min_entities: int = 2,
    ):
        """
        Args:
            confidence_thresholds: Override confidence thresholds per entity type.
            min_entities: Minimum number of entities required for a valid query.
        """
        self.confidence_thresholds = confidence_thresholds or CONFIDENCE_THRESHOLDS
        self.min_entities = min_entities

    def build_query(self, entities: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Build a cleaned geocoding query from extracted entities.

        Args:
            entities: Dict of entity_type -> {text, confidence, ...}

        Returns:
            Dict with:
                - 'query_string': Cleaned query for geocoding
                - 'components': Ordered list of included components
                - 'confidence': Overall query confidence
                - 'is_valid': Whether the query meets minimum requirements
                - 'missing_components': Entity types with low confidence
        """
        # Filter entities by confidence
        valid_entities = {}
        missing_components = []

        for entity_type in ENTITY_PRIORITY:
            if entity_type in entities:
                entity = entities[entity_type]
                threshold = self.confidence_thresholds.get(entity_type, 0.5)
                conf = entity.get("confidence", 0)

                if conf >= threshold:
                    valid_entities[entity_type] = entity
                else:
                    missing_components.append({
                        "type": entity_type,
                        "confidence": conf,
                        "threshold": threshold,
                    })

        # Build query string in hierarchical order (most specific to least)
        # For geocoding, we reverse: City > Locality > Landmark > Street
        query_parts = []
        components = []

        # Build in geocoding-friendly order
        geocode_order = ["HOUSE_NO", "BUILDING", "STREET", "LANDMARK", "LOCALITY", "CITY", "PINCODE"]

        for entity_type in geocode_order:
            if entity_type in valid_entities:
                text = valid_entities[entity_type]["text"]
                query_parts.append(text)
                components.append({
                    "type": entity_type,
                    "text": text,
                    "confidence": valid_entities[entity_type].get("confidence", 0),
                })

        # Also include state if present
        if "STATE" in entities:
            state = entities["STATE"]
            threshold = self.confidence_thresholds.get("STATE", 0.4)
            if state.get("confidence", 0) >= threshold:
                query_parts.append(state["text"])
                components.append({
                    "type": "STATE",
                    "text": state["text"],
                    "confidence": state.get("confidence", 0),
                })

        query_string = ", ".join(query_parts)

        # Calculate overall confidence
        if components:
            overall_confidence = sum(c["confidence"] for c in components) / len(components)
        else:
            overall_confidence = 0.0

        # Check validity
        has_area = any(c["type"] in ("CITY", "LOCALITY", "PINCODE") for c in components)
        is_valid = len(components) >= self.min_entities and has_area

        return {
            "query_string": query_string,
            "components": components,
            "confidence": round(overall_confidence, 4),
            "is_valid": is_valid,
            "missing_components": missing_components,
            "num_entities": len(components),
        }
