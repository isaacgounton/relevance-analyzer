# Enhanced Relevance Analyzer v2.0

Next-generation AI API for analyzing embed relevance to articles using adaptive intelligence, contextual understanding, and dynamic threshold optimization.

## 🚀 Major Enhancements v2.0

- **Dynamic Threshold Calculation** - Article complexity scoring with adaptive thresholds (no more static values!)
- **Contextual Entity Matching** - Weighted importance, partial matching, and hierarchical relationships
- **Temporal Relevance Detection** - Time-sensitive content boosting for breaking news and timely embeds
- **Geographic Relevance** - Location-based analysis with hierarchical matching (e.g., "San Francisco" matches "California")
- **Enhanced Semantic Analysis** - Concept extraction with focused similarity scoring
- **Domain-Specific Optimization** - Tailored algorithms for news, tech, business, medical, and legal content
- **Multi-dimensional Scoring** - 50% enhanced semantic, 30% contextual entity, 20% contextual analysis

## Installation

```bash
git clone <repository>
cd relevance-analyzer
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Quick Start

```bash

# Or original version
python main.py
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

**Enhanced Response v2.0:**
```json
{
  "keep": true,
  "reason": "Strong conceptual alignment, high entity precision, temporal relevance (+0.12) | cross-reference (+0.18)",
  "scores": {
    "enhanced_semantic": 0.724,
    "entity_similarity": 0.683,
    "contextual_score": 0.342,
    "article_complexity": 0.721,
    "entity_richness": 12,
    "final_score": 0.674,
    "threshold": 0.158,
    "temporal_boost": 0.120,
    "geographic_boost": 0.000
  }
}
```

### 🎯 Key Improvements in v2.0

**Dynamic Threshold Example:**
- **Complex Article** (high entity density): `threshold: 0.158` (more permissive)
- **Simple Article** (low entity density): `threshold: 0.285` (stricter filtering)
- **No more static 0.2 threshold!**

### POST /explain-relevance

Returns detailed AI-powered explanation of the relevance analysis with scoring breakdown and recommendations.

### POST /screenshot-embeds

Detect and capture screenshots of embeds within an article.

**Request:**
```json
{
  "url": "https://www.monfric.ca/nouvelles/la-banque-du-canada-annonce-sa-decision-concernant-le-taux-directeur-du-10-decembre",
  "embed_url": "https://x.com/fordnation/status/1978779503213052337" (optional)
}
```

**Response:**
```json
{
  "url": "https://www.monfric.ca/...",
  "screenshots": [
    {
      "type": "twitter",
      "index": 0,
      "screenshot": "base64_encoded_image..."
    }
  ],
  "count": 1
}
```

### POST /screenshot-single-embed

Directly screenshot a single embed URL (social post, video, etc.).

**Request:**
```json
{
  "url": "https://twitter.com/fordnation/status/1978779503213052337"
}
```

**Response:**
```json
{
  "url": "https://twitter.com/...",
  "type": "twitter",
  "screenshot": "base64_encoded_image..."
}
```

### GET /config

Returns available configuration options and usage guidelines.

## Configuration Options

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `threshold` | float | **dynamic** | 0.0-1.0 | Custom threshold override (disables dynamic calculation) |
| `strictness` | string | `"standard"` | `"strict"`, `"standard"`, `"lenient"` | Filter sensitivity (±0.08 adjustment) |
| `domain` | string | `"general"` | `"news"`, `"tech"`, `"business"`, `"medical"`, `"legal"` | Content type optimization |

### 🔄 Dynamic Threshold Algorithm v2.0

**Article Complexity Factors:**
- Entity density (40%): Number of entities per word
- Lexical diversity (30%): Unique words / total words
- Semantic diversity (30%): Sentence length variance

**Threshold Calculation:**
- **Complex article** (>0.7): `0.15` base threshold
- **Medium article** (0.4-0.7): `0.20` base threshold
- **Simple article** (<0.4): `0.25` base threshold

**Adjustments:**
- Entity richness: ±0.05 based on entity count
- Embed length: ±0.05 to ±0.10 based on length
- Strictness: ±0.08 for strict/lenient modes

## Performance Metrics

- **Latency**: ~50ms per analysis (CPU)
- **Memory**: ~500MB RAM
- **Throughput**: 20+ requests/second
- **Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Languages**: 100+ supported

## 🧠 Enhanced AI Scoring Components v2.0

**Enhanced Semantic Analysis (50%)**
- Transformer-based meaning comprehension
- Concept extraction and focused similarity
- Keyword-level topical alignment
- Multilingual semantic understanding

**Contextual Entity Matching (30%)**
- Weighted entity importance (ORG/PERSON = 1.0, GPE = 0.7, DATE = 0.4)
- Partial fuzzy matching (70% similarity threshold)
- Hierarchical relationships (San Francisco → California)
- Entity type prioritization

**Contextual Analysis (20%)**
- **Temporal relevance** (0.0-0.2): Time-sensitive content boost
- **Geographic relevance** (0.0-0.3): Location-based matching
- **Cross-reference analysis** (0.0-0.2): Entity mention detection
- **Authority indicators** (0.0-0.15): Credibility assessment

### 📊 Domain-Specific Boosts

**News Domain**: +50% temporal boost, +30% geographic boost
**Tech Domain**: +20% entity precision boost
**Business Domain**: +10% organizational entity boost

## 🎯 Advanced Features v2.0

### Entity Intelligence
- **13 Entity Types**: ORG, PERSON, GPE, EVENT, PRODUCT, WORK_OF_ART, LAW, DATE, FAC, LOC, MONEY, PERCENT
- **Importance Weighting**: Critical entities (ORG/PERSON) weighted higher than dates/locations
- **Partial Matching**: "Fed" matches "Federal Reserve" with 70% similarity threshold
- **Hierarchical Detection**: "San Francisco" connects to "California" automatically

### Temporal Intelligence
- **Time Pattern Recognition**: Breaking news, recent developments, real-time updates
- **Date Extraction**: Multiple date formats and temporal expressions
- **Timeliness Scoring**: Fresh content gets higher relevance scores
- **News Cycle Awareness**: Prioritizes current events over historical references

### Geographic Intelligence
- **Location Hierarchy**: City → State → Country relationships
- **Regional Relevance**: Local news prioritized for specific locations
- **Cross-Border Matching**: Handles international location references
- **Proximity Scoring**: Geographic distance affects relevance scores

### Contextual Authority Detection
- **Semantic Indicators**: "official", "confirmed", "announced", "verified"
- **Source Assessment**: Credibility scoring based on language patterns
- **Domain-Specific Patterns**: Different authority signals for news vs tech vs business

## 🔐 Session Persistence (Bypassing Login Walls)

If Meta (Facebook/Instagram) or other platforms block screenshots with a login page, you can provide a pre-authenticated session.

### 1. Generate your session file
Run the helper script locally on your machine (requires a screen/GUI):

```bash
python scripts/generate_auth.py
```

1. A browser window will open.
2. Log into Facebook and Instagram.
3. Once logged in, go back to the terminal and press **ENTER**.
4. This creates an `auth.json` file in your root directory.

### 2. Use with the API
Ensure `auth.json` is present in the same directory where you run `main.py`. The `ScreenshotService` will automatically detect it and use your active session to bypass login prompts.

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


---

## 🔥 Version Comparison

| Feature | v1.0 | v2.0 (Enhanced) |
|---------|------|-----------------|
| **Threshold** | Static (0.2 for standard) | Dynamic based on article complexity |
| **Entity Matching** | Simple Jaccard similarity | Weighted + partial + hierarchical matching |
| **Semantic Analysis** | Basic cosine similarity | Concept extraction + focused scoring |
| **Temporal Awareness** | None | Time-sensitive content boosting |
| **Geographic Awareness** | None | Location hierarchy + proximity scoring |
| **Domain Optimization** | Basic authority patterns | Advanced domain-specific algorithms |
| **Scoring Weights** | 60%/25%/15% | 50%/30%/20% with contextual boosts |
| **Accuracy** | Good | **Significantly Improved** |

## 🚀 Why v2.0 is Revolutionary

**Before**: Static threshold meant the same relevance criteria for simple news articles and complex financial reports.

**After**: Dynamic thresholds adapt to content characteristics - complex articles get more permissive thresholds (easier to match), while simple articles get stricter filtering (higher quality control).

**Real-World Impact**:
- **News Organizations**: Better filtering for breaking news vs feature articles
- **Content Platforms**: More accurate embed recommendations by content type
- **Social Media**: Improved relevance detection for trending vs evergreen content

**This enhanced API provides a next-generation, adaptive solution for content relevance analysis that intelligently adjusts to the complexity and characteristics of each article.**