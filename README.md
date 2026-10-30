# 🗺️ GeoParse-India

[![Tests](https://github.com/susmitgoswami5/GeoParse-India/actions/workflows/tests.yml/badge.svg)](https://github.com/susmitgoswami5/GeoParse-India/actions/workflows/tests.yml)

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

### Train the NER Model

```bash
python -m geoparse.ner.trainer --train-data data/train.json --val-data data/val.json --epochs 5
```

Trained weights are not committed (they are ~250 MB). The trainer reports entity-level precision, recall and F1 (seqeval) on the validation set. **If no trained model is found, the API falls back to rule-based extraction** (PIN-code regex + dictionary lookup), so the demo runs without a GPU, but the NER component is then not being exercised.

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

## 📊 Evaluation

### Phonetic correction layer

Reproduce with `python -m evaluation.evaluate_phonetic` (results in `evaluation/phonetic_results.json`).

| Test set | Original | With false-correction guard |
|---|---|---|
| **A.** 285 synthetic misspellings of the 24 covered cities (5 noise types, independent of the hand-written variant table) | 99.6% corrected | 99.6% corrected |
| **B.** 110 real Indian cities outside coverage, known to the guard | 12 wrongly rewritten | 0 wrongly rewritten |
| **C.** 60 real Indian cities in *neither* dictionary (honest held-out test) | 6 wrongly rewritten (10%) | 3 wrongly rewritten (5%) |

**Finding.** The original corrector silently rewrote real cities into similar-looking covered ones, e.g. *Raipur → Jaipur*, *Kanpur → Nagpur*, *Satna → Patna*. For a geocoder this is worse than no correction, because the address is confidently placed in the wrong city. Two changes fix most of it:

1. `OTHER_INDIAN_CITIES`: real cities outside coverage are recognised and left unchanged, so the geocoder flags them for review instead of mis-locating them.
2. **First-letter guard:** a fuzzy match that changes the first letter is accepted only at Jaro–Winkler ≥ 0.95. Real misspellings and abbreviations ("Bnglr", "Hydrabad") almost always keep the first letter.

The remaining errors share a first letter with a covered city (*Kollam → Kolkata*, *Ahmednagar → Ahmedabad*, *Purnia → Pune*). Fixing these needs a full national gazetteer or context from the PIN code (see *Limitations*).

### NER model

Entity-level F1 is reported by the trainer on the held-out validation split. The data is synthetic, so a high F1 shows the model learns the generator's patterns, not that it generalises to real, messy addresses. The next step is a small hand-labelled set of real addresses.

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
├── evaluation/         # Reproducible evaluation scripts + results
├── tests/              # Unit & integration tests (incl. false-correction guard)
├── pyproject.toml      # Project configuration
└── README.md
```

---

## ⚠️ Limitations & Next Steps

- **Synthetic training data.** Both the NER training data and the correction tests are generated, so real-world accuracy is unknown until the model is tested on hand-labelled real addresses.
- **Small coverage.** The geocoder covers 24 cities with representative coordinates and PIN codes; it is a prototype, not a national gazetteer.
- **PIN-code consistency.** The corrector does not yet use the PIN code to disambiguate the city. That would resolve most remaining false corrections (e.g. a 691xxx PIN means Kollam, Kerala, not Kolkata).
- **Locality correction** uses Jaro–Winkler without the first-letter guard. The same evaluation should be extended to localities.

## 👤 Author

Susmit Goswami — B.Tech CSE (AI & ML), Techno Main Salt Lake, Kolkata · [GitHub](https://github.com/susmitgoswami5) · [LinkedIn](https://www.linkedin.com/in/susmit-goswami-33784542a/)

## 📜 License

MIT
