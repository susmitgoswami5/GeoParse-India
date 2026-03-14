"""
Pydantic request/response schemas for the GeoParse-India API.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---- Request Models ----

class ParseRequest(BaseModel):
    """Request body for address parsing."""
    address: str = Field(..., min_length=5, max_length=500, description="Raw address text to parse")


class GeocodeRequest(BaseModel):
    """Request body for full geocoding pipeline."""
    address: str = Field(..., min_length=5, max_length=500, description="Raw address text to geocode")


class BatchGeocodeRequest(BaseModel):
    """Request body for batch geocoding."""
    addresses: List[str] = Field(..., min_length=1, max_length=100, description="List of address strings")


# ---- Response Models ----

class EntityDetail(BaseModel):
    """Detail of a single extracted entity."""
    text: str
    confidence: float
    original: Optional[str] = None
    was_corrected: bool = False
    correction_confidence: Optional[float] = None


class TokenPrediction(BaseModel):
    """Per-token NER prediction."""
    token: str
    label: str
    confidence: float


class QueryComponent(BaseModel):
    """A component of the cleaned geocoding query."""
    type: str
    text: str
    confidence: float


class QueryResult(BaseModel):
    """Result of query building."""
    query_string: str
    components: List[QueryComponent]
    confidence: float
    is_valid: bool
    num_entities: int


class ParseResponse(BaseModel):
    """Response for address parsing."""
    entities: Dict[str, EntityDetail]
    token_predictions: List[TokenPrediction]
    input_text: str


class GeocodeResponse(BaseModel):
    """Response for full geocoding."""
    lat: Optional[float] = None
    lng: Optional[float] = None
    confidence: float = 0.0
    radius_m: Optional[float] = None
    resolution: str = "none"
    needs_review: bool = False
    review_reason: Optional[str] = None
    h3_cell: Optional[str] = None
    entities: Dict[str, Any] = {}
    raw_entities: Dict[str, Any] = {}
    query: Dict[str, Any] = {}
    input_text: str = ""
    token_predictions: List[Dict[str, Any]] = []


class BatchGeocodeResponse(BaseModel):
    """Response for batch geocoding."""
    results: List[GeocodeResponse]
    total: int
    successful: int
    needs_review: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "1.0.0"
    model_loaded: bool = False
    components: Dict[str, str] = {}
