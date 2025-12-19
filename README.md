# Improved Embed Relevance Analyzer

A FastAPI-based API that uses **semantic AI understanding** to analyze if an embed (video, social media post, etc.) is relevant to an article. This new version significantly outperforms traditional keyword-based approaches by understanding meaning and context.

## 🚀 What's New

- **🧠 Semantic Understanding**: Uses sentence transformers for true meaning-based analysis
- **🌍 Multilingual Support**: Excellent French/English comprehension
- **🏛️ Source Quality Detection**: Identifies official and authoritative sources
- **⚡ Entity Recognition**: Understands people, organizations, and locations
- **🎯 Smart Scoring**: Adaptive thresholds based on content characteristics

## Key Improvements Over Traditional Approaches

| Traditional Approach | Semantic AI Approach |
|---------------------|---------------------|
| ❌ "Tweet officiel" vs "Banque du Canada" = Low match | ✅ Understands both refer to same entity |
| ❌ "interest rate" vs "taux directeur" = Different terms | ✅ Knows these are synonyms |
| ❌ Only counts word overlaps | ✅ Understands context and meaning |
| ❌ No source quality assessment | ✅ Recognizes official sources |

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

### Start the API

```bash
python relevance_api.py
```

The API will be available at `http://localhost:8000`

### Endpoints

#### POST /analyze-relevance

Analyzes if an embed is relevant to an article using semantic understanding.

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

Provides detailed AI-powered explanation of the analysis.

#### GET /health

Health check endpoint.

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_api.py
```

This will test scenarios that Claude handled correctly but the old analyzer rejected:

- ✅ Official Bank of Canada announcements
- ✅ Trade relation comments from political figures
- ✅ Related economic analysis content
- ❌ Random celebrity gossip (correctly rejected)

## 🎯 Real-World Performance

### Before (Traditional Approach)
```
❌ EMBED REJECTED - This embed does not appear relevant to the article.
Overall Score: 0.092/1.000 (threshold: 0.250)
🔍 Content Similarity Analysis: Very low semantic similarity (0.008)
🏷️ Keyword Overlap Analysis: Found 4 shared keywords (0.017 similarity)
```

### After (Semantic AI Approach)
```
✅ EMBED KEPT - This embed appears relevant to the article.
Overall Score: 0.688/1.000 (threshold: 0.250)
🧠 Semantic Understanding Analysis: High semantic similarity (0.842)
🏛️ Source Quality Analysis: Official source detected (+0.20)
```

## Integration with n8n

Replace your Claude node with an HTTP Request node calling:

- **URL**: `http://localhost:8000/analyze-relevance`
- **Method**: POST
- **Body**: JSON with article and embed data

This gives you Claude-like semantic understanding at a fraction of the cost and latency!

## 🔧 How It Works

1. **Semantic Similarity (60% weight)**: Uses multilingual sentence transformers to understand meaning
2. **Entity Overlap (25% weight)**: Identifies shared people, organizations, locations
3. **Source Boost (15% weight)**: Rewards official sources and relevant topic mentions
4. **Adaptive Thresholds**: Adjusts requirements based on content length

## 📊 Model Details

- **Primary Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Language Support**: Excellent French/English bilingual comprehension
- **Speed**: ~50ms per analysis on CPU
- **Memory**: ~500MB model size

## Development Setup

### VS Code Configuration

The project includes VS Code settings (`.vscode/settings.json`) that automatically configure the Python interpreter to use the virtual environment. This resolves import errors in Pylance.

**Note**: If you see Pylance errors about missing imports, make sure VS Code is using the virtual environment. You can check this in the status bar at the bottom of VS Code - it should show "Python 3.x.x ('venv': venv)".

If you're not using VS Code, make sure your Python IDE is configured to use the virtual environment at `./venv/bin/python`.

### Docker Deployment

1. Build the image:
```bash
docker build -t relevance-analyzer .
```

2. Run the container:
```bash
docker run -p 8000:8000 relevance-analyzer
```

### Production Deployment

For production, consider using:
- **Railway**: `railway deploy`
- **Render**: Connect your GitHub repo
- **Heroku**: `heroku create && git push heroku main`
- **Vercel**: For serverless deployment
- **AWS/GCP/Azure**: Container services

## API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI).