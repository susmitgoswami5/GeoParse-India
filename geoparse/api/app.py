"""
FastAPI application for GeoParse-India.

Provides endpoints for:
- POST /api/parse — Parse address text → extracted entities
- POST /api/geocode — Full pipeline: parse + correct + geocode
- POST /api/batch — Batch processing
- GET /api/health — Health check

Serves the static web UI from geoparse/static/.
"""

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from geoparse.api.models import (
    BatchGeocodeRequest,
    BatchGeocodeResponse,
    GeocodeRequest,
    GeocodeResponse,
    HealthResponse,
    ParseRequest,
    ParseResponse,
)
from geoparse.geocoder.engine import GeocodingEngine
from geoparse.phonetic.corrector import PhoneticCorrector
from geoparse.geocoder.query_builder import QueryBuilder
from geoparse.geocoder.spatial_index import SpatialIndex


# ---- Initialize components ----

# Try to load NER model if available
ner_parser = None
model_path = os.environ.get("GEOPARSE_MODEL_PATH", "models/address_ner/best")

if Path(model_path).exists():
    try:
        from geoparse.ner.inference import AddressNERParser
        ner_parser = AddressNERParser(model_path=model_path)
        print(f"✅ NER model loaded from {model_path}")
    except Exception as e:
        print(f"⚠️  Could not load NER model: {e}")
        print("   Using rule-based extraction as fallback.")
else:
    print(f"ℹ️  No NER model found at {model_path}")
    print("   Using rule-based extraction. Train a model with:")
    print("   python -m geoparse.ner.trainer")

# Initialize pipeline components
phonetic_corrector = PhoneticCorrector()
query_builder = QueryBuilder()
spatial_index = SpatialIndex()
geocoding_engine = GeocodingEngine(
    ner_parser=ner_parser,
    phonetic_corrector=phonetic_corrector,
    query_builder=query_builder,
    spatial_index=spatial_index,
)

# ---- FastAPI App ----

app = FastAPI(
    title="GeoParse-India",
    description="Hierarchical Address Disambiguation & Geocoding Engine for Indian Addresses",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- API Endpoints ----

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=ner_parser is not None,
        components={
            "ner": "transformer" if ner_parser else "rule-based",
            "phonetic": "active",
            "spatial": "h3-indexed",
            "geocoder": "active",
        },
    )


@app.post("/api/parse", response_model=ParseResponse)
async def parse_address(request: ParseRequest):
    """
    Parse an address and extract named entities.

    Returns hierarchical entities (house number, building, street,
    landmark, locality, city, state, pincode) with confidence scores.
    """
    try:
        result = geocoding_engine.geocode(request.address)

        entities = {}
        for k, v in result.entities.items():
            entities[k] = {
                "text": v.get("text", ""),
                "confidence": v.get("confidence", 0),
                "original": v.get("original"),
                "was_corrected": v.get("was_corrected", False),
                "correction_confidence": v.get("correction_confidence"),
            }

        return ParseResponse(
            entities=entities,
            token_predictions=result.token_predictions,
            input_text=result.input_text,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parse error: {str(e)}")


@app.post("/api/geocode", response_model=GeocodeResponse)
async def geocode_address(request: GeocodeRequest):
    """
    Full geocoding pipeline: parse + phonetic correction + geocode.

    Returns lat/lng, confidence circle, entity breakdown, and
    manual review flag if address is ambiguous.
    """
    try:
        result = geocoding_engine.geocode(request.address)
        return GeocodeResponse(**result.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Geocode error: {str(e)}")


@app.post("/api/batch", response_model=BatchGeocodeResponse)
async def batch_geocode(request: BatchGeocodeRequest):
    """Batch geocode multiple addresses."""
    try:
        results = geocoding_engine.geocode_batch(request.addresses)
        responses = [GeocodeResponse(**r.to_dict()) for r in results]

        successful = sum(1 for r in responses if r.lat is not None)
        needs_review = sum(1 for r in responses if r.needs_review)

        return BatchGeocodeResponse(
            results=responses,
            total=len(responses),
            successful=successful,
            needs_review=needs_review,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch error: {str(e)}")


# ---- Static file serving (Web UI) ----

static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def serve_ui():
    """Serve the web UI."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "GeoParse-India API", "docs": "/docs"}


# ---- Run ----

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
