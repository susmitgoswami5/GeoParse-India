"""
Integration tests for the FastAPI endpoints.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from geoparse.api.app import app


@pytest.fixture
def transport():
    return ASGITransport(app=app)


@pytest.mark.asyncio
async def test_health_endpoint(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "components" in data


@pytest.mark.asyncio
async def test_parse_endpoint(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/parse", json={
            "address": "Near SBI Bank, Koramangala, Bengaluru 560034"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "entities" in data
        assert "input_text" in data


@pytest.mark.asyncio
async def test_geocode_endpoint(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/geocode", json={
            "address": "42/B MG Road, Koramangala, Bengaluru 560034"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "lat" in data
        assert "lng" in data
        assert "confidence" in data
        assert "entities" in data


@pytest.mark.asyncio
async def test_geocode_returns_location(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/geocode", json={
            "address": "Whitefield, Bangalore 560066"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["lat"] is not None
        assert data["lng"] is not None
        assert data["radius_m"] is not None


@pytest.mark.asyncio
async def test_batch_endpoint(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/batch", json={
            "addresses": [
                "Koramangala, Bengaluru",
                "Connaught Place, Delhi 110001",
            ]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2


@pytest.mark.asyncio
async def test_parse_validation_error(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/parse", json={
            "address": "Hi"  # Too short (min_length=5)
        })
        assert resp.status_code == 422


@pytest.mark.asyncio
async def test_serve_ui(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
        assert resp.status_code == 200
