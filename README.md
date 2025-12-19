# Relevance Analyzer

A FastAPI-based API that analyzes the relevance of embeds (videos, social media posts, etc.) to articles using semantic AI understanding.

## Features

- Semantic content analysis using sentence transformers
- Multilingual support (French/English)
- Entity recognition for people, organizations, and locations
- Source quality assessment
- Adaptive scoring thresholds

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

### Starting the API

```bash
python relevance_api.py
```

The API will be available at `http://localhost:8000`.

### API Endpoints

#### POST /analyze-relevance

Analyzes the relevance of an embed to an article.

**Request Body:**
```json
{
  "articleTitle": "Banque du Canada maintient son taux directeur",
  "articleSummary": "La Banque du Canada a décidé de maintenir son taux directeur à 5%.",
  "embedType": "tweet",
  "embedContent": "Tweet officiel de la Banque du Canada confirmant exactement le contenu de l'article sur le maintien du taux directeur",
  "articleText": "La Banque du Canada maintient son taux directeur à 5%..."
}
```

**Response:**
```json
{
  "keep": true,
  "reason": "Strong semantic similarity, high entity overlap, source boost: 0.20",
  "scores": {
    "semantic_similarity": 0.842,
    "entity_overlap": 0.667,
    "source_boost": 0.200,
    "final_score": 0.688,
    "threshold": 0.250
  }
}
```

#### POST /explain-relevance

Provides detailed explanation of the relevance analysis.

#### GET /health

Health check endpoint.

## Testing

Run the test suite:

```bash
python test_api.py
```

## How It Works

The analyzer combines multiple scoring components:

- **Semantic Similarity** (60%): Uses sentence transformers to understand content meaning
- **Entity Overlap** (25%): Identifies shared entities (people, organizations, locations)
- **Source Quality** (15%): Boosts scores for official and authoritative sources

## Model Details

- **Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Languages**: French and English
- **Performance**: ~50ms per analysis on CPU
- **Memory**: ~500MB

## Deployment

### Docker

```bash
docker build -t relevance-analyzer .
docker run -p 8000:8000 relevance-analyzer
```

### Production Platforms

- Railway
- Render
- Heroku
- Vercel
- AWS/GCP/Azure container services

## API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.