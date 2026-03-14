# 🗺️ GeoParse-India

**Hierarchical Address Disambiguation & Geocoding Engine for Indian Addresses**

GeoParse-India is a specialized NER + geocoding pipeline tailored for unstructured Indian addresses. It parses raw text to extract hierarchical components (House No, Building, Street, Landmark, Locality, City, Pincode), applies phonetic correction for transliteration variants, and geocodes the cleaned address with confidence scoring.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Raw Address Input                       │
│  "Behind banyan tree, nr Sharma Medicals, Whitefield,  │
│   Bnglr 560066"                                         │
└──────────────────────┬──────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │   NER Engine    │  Token Classification
              │  (DistilBERT)   │  B-LANDMARK, B-LOCALITY, etc.
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │    Phonetic     │  "Bnglr" → "Bengaluru"
              │   Corrector     │  Double Metaphone + Jaro-Winkler
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  Query Builder  │  Cleaned: "Whitefield, Bengaluru, 560066"
              │  (Confidence)   │  Priority: Pincode > City > Locality
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  Spatial Index  │  H3 Hexagonal Lookup
              │    (H3/Uber)    │  → Lat/Lng + Confidence Circle
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  GeocodeResult  │  {lat, lng, radius, confidence,
              │                 │   needs_review, entities}
              └─────────────────┘
```

---

## 🚀 Quick Start

### Install

```bash
cd GeoParse-India
pip install -e ".[dev]"
```

### Generate Training Data

```bash
python -m geoparse.data.generate_dataset --num-samples 50000 --output-dir data/
```

### Train NER Model (Optional)

```bash
python -m geoparse.ner.trainer --train-data data/train.json --val-data data/val.json --epochs 5
```

### Start the Server

```bash
uvicorn geoparse.api.app:app --reload --port 8000
```

Then open **http://localhost:8000** for the web UI.

### Run Tests

```bash
pytest tests/ -v
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | System health check |
| `/api/parse` | POST | Parse address → extract entities |
| `/api/geocode` | POST | Full pipeline: parse + correct + geocode |
| `/api/batch` | POST | Batch geocode multiple addresses |
| `/docs` | GET | Interactive Swagger UI |

### Example Request

```bash
curl -X POST http://localhost:8000/api/geocode \
  -H "Content-Type: application/json" \
  -d '{"address": "Near SBI Bank, Koramangala, Bangalore 560034"}'
```

---

## 🧩 Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| NER Model | DistilBERT (HuggingFace) | Token classification for address entities |
| Phonetic Corrector | Double Metaphone + Jaro-Winkler | Handles transliteration variants |
| Spatial Index | H3 (Uber) | Hexagonal geo-indexing for locality lookup |
| API | FastAPI | High-performance REST API |
| Web UI | HTML/CSS/JS + Leaflet | Dark glassmorphism demo interface |
| Data | Synthetic Generator | BIO-tagged training data from component dictionaries |

---

## 📁 Project Structure

```
GeoParse-India/
├── geoparse/
│   ├── api/            # FastAPI service
│   │   ├── app.py      # Main application
│   │   └── models.py   # Pydantic schemas
│   ├── data/           # Synthetic data factory
│   │   ├── address_components.py  # Indian address dictionaries
│   │   ├── synthetic_generator.py # Noisy address generator
│   │   └── generate_dataset.py    # CLI for dataset generation
│   ├── geocoder/       # Geocoding heuristic engine
│   │   ├── engine.py   # Full pipeline orchestrator
│   │   ├── query_builder.py  # Cleaned query construction
│   │   └── spatial_index.py  # H3-based spatial lookup
│   ├── ner/            # Hierarchical NER model
│   │   ├── dataset.py  # PyTorch Dataset with sub-word alignment
│   │   ├── inference.py # Entity extraction from trained model
│   │   ├── label_schema.py # BIO label definitions
│   │   └── trainer.py  # Fine-tuning script
│   ├── phonetic/       # Phonetic correction layer
│   │   └── corrector.py # Double Metaphone + Jaro-Winkler
│   └── static/         # Web UI
│       ├── index.html  # Premium dark glassmorphism interface
│       ├── style.css   # Design system
│       └── app.js      # Frontend logic + Leaflet map
├── tests/              # Unit & integration tests
├── pyproject.toml      # Project configuration
└── README.md
```

---

## 📜 License

MIT
