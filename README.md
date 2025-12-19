# Relevance Analyzer

Enterprise-grade AI API for analyzing embed relevance to articles using advanced semantic understanding and multi-dimensional scoring.

## Core Capabilities

- **Advanced Semantic Analysis** - Sentence transformer-based meaning comprehension across 100+ languages
- **Comprehensive Entity Recognition** - 13 entity types including organizations, people, locations, events, products, and legal entities
- **Authority Pattern Detection** - 80+ authoritative communication patterns for news, business, technical, and official content
- **Configurable Thresholds** - Customizable strictness levels and domain-specific optimization
- **Multilingual Support** - Built-in multilingual authority patterns and semantic models

## Installation

```bash
git clone <repository>
cd relevance-analyzer
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Quick Start

```bash
python relevance_api.py
```

API available at `http://localhost:8000` • Interactive docs at `http://localhost:8000/docs`

## API Endpoints

### POST /analyze-relevance

**Request:**
```json
{
  "articleTitle": "Federal Reserve Announces Interest Rate Decision",
  "articleSummary": "The Fed decided to maintain current interest rates amid economic uncertainty.",
  "embedType": "social_post",
  "embedContent": "BREAKING: Federal Reserve confirms interest rates remain unchanged at 5.25% according to official sources",
  "articleText": "Full article text...",
  "threshold": 0.25,
  "strictness": "standard",
  "domain": "news"
}
```

**Response:**
```json
{
  "keep": true,
  "reason": "Good semantic similarity, high entity overlap, content boost: 0.18",
  "scores": {
    "semantic_similarity": 0.742,
    "entity_overlap": 0.314,
    "content_boost": 0.180,
    "final_score": 0.532,
    "threshold": 0.250
  }
}
```

### POST /explain-relevance

Returns detailed AI-powered explanation of the relevance analysis with scoring breakdown and recommendations.

### GET /config

Returns available configuration options and usage guidelines.

## Configuration Options

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `threshold` | float | adaptive | 0.0-1.0 | Custom threshold override |
| `strictness` | string | `"standard"` | `"strict"`, `"standard"`, `"lenient"` | Filter sensitivity |
| `domain` | string | `"general"` | `"news"`, `"tech"`, `"business"`, `"medical"`, `"legal"` | Content type optimization |

## Performance Metrics

- **Latency**: ~50ms per analysis (CPU)
- **Memory**: ~500MB RAM
- **Throughput**: 20+ requests/second
- **Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Languages**: 100+ supported

## AI Scoring Components

**Semantic Similarity (60%)** - Transformer-based meaning comprehension and contextual understanding

**Entity Overlap (25%)** - Named entity recognition across 13 types: ORG, PERSON, GPE, EVENT, PRODUCT, WORK_OF_ART, LAW, DATE, FAC, LOC, MONEY, PERCENT

**Content Quality (15%)** - Authority pattern detection, cross-reference analysis, and topic coherence assessment

## Authority Pattern Categories

- **News & Reporting**: Breaking news, eyewitness reports, official announcements
- **Corporate Communications**: Financial results, strategic initiatives, executive decisions
- **Technical & Scientific**: Research findings, peer-reviewed studies, clinical trials
- **Legal & Regulatory**: Court rulings, legislation, compliance documentation
- **Emergency Alerts**: Public safety, weather warnings, health advisories
- **Multilingual**: Authority patterns in 6 major languages

## Deployment

### Docker
```bash
docker build -t relevance-analyzer .
docker run -p 8000:8000 relevance-analyzer
```

### Production Platforms
- Railway • Render • Vercel • AWS/GCP/Azure • Heroku

### Environment Variables
```bash
PORT=8000
WORKERS=4
LOG_LEVEL=info
```

## API Documentation

Interactive Swagger UI available at `http://localhost:8000/docs` with request/response examples and testing capabilities.

## Testing

```bash
python test_api.py
```

Comprehensive test suite covering accuracy, performance, and edge cases.

---


This API provides a robust, AI-powered solution for content relevance analysis with configurable parameters and detailed explanations.