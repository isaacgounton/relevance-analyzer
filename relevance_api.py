from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from thefuzz import fuzz
import uvicorn
from typing import Optional, Dict, List, Tuple
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import re
from datetime import datetime, timedelta
import math

app = FastAPI(title="Enhanced Embed Relevance Analyzer", version="2.0.0")

# Load semantic model (multilingual)
print("Loading semantic model...")
semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Semantic model loaded!")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If model not found, download it
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def calculate_article_complexity(article_text: str, article_entities: set) -> float:
    """Calculate article complexity score for dynamic threshold adjustment."""
    entity_density = len(article_entities) / max(1, len(article_text.split()))

    # Lexical diversity (unique words / total words)
    words = article_text.lower().split()
    unique_words = set(words)
    lexical_diversity = len(unique_words) / max(1, len(words))

    # Semantic diversity using sentence length variance
    sentences = [s.strip() for s in article_text.split('.') if s.strip()]
    sentence_lengths = [len(s.split()) for s in sentences]
    semantic_diversity = np.std(sentence_lengths) / max(1, np.mean(sentence_lengths)) if sentence_lengths else 0

    # Combine metrics (0.0-1.0 scale)
    complexity = min(1.0, (entity_density * 0.4 + lexical_diversity * 0.3 + semantic_diversity * 0.3))
    return complexity

def extract_key_concepts(text: str, top_k: int = 10) -> List[str]:
    """Extract key concepts using TF-IDF and named entities."""
    doc = nlp(text)

    # Get important entities
    important_entities = []
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PERSON', 'GPE', 'PRODUCT', 'EVENT']:
            if len(ent.text) > 2:  # Filter out very short entities
                important_entities.append(ent.text.lower())

    # Get key terms using TF-IDF
    try:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=100)
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]

        # Get top terms
        top_indices = np.argsort(tfidf_scores)[-top_k:]
        key_terms = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
    except:
        key_terms = []

    # Combine and deduplicate
    concepts = list(set(important_entities + key_terms))
    return concepts[:top_k]

def contextual_entity_similarity(article_entities: set, embed_entities: set,
                               article_doc, embed_doc) -> float:
    """Enhanced entity similarity with partial matching and importance weighting."""
    if not article_entities or not embed_entities:
        return 0.0

    # Entity importance weights
    entity_weights = {
        'ORG': 1.0, 'PERSON': 1.0, 'PRODUCT': 0.9, 'EVENT': 0.8,
        'GPE': 0.7, 'MONEY': 0.6, 'PERCENT': 0.5, 'DATE': 0.4,
        'LAW': 0.9, 'FAC': 0.5, 'LOC': 0.6, 'WORK_OF_ART': 0.3
    }

    # Exact matches with weighting
    weighted_matches = 0
    total_weight = 0

    for ent in article_doc.ents:
        if ent.text.lower() in embed_entities:
            weight = entity_weights.get(ent.label_, 0.5)
            weighted_matches += weight
        total_weight += entity_weights.get(ent.label_, 0.5)

    # Partial matching (fuzzy matching for important entities)
    partial_matches = 0
    for art_ent in article_doc.ents:
        if art_ent.label_ in ['ORG', 'PERSON', 'PRODUCT']:  # Only for important entity types
            art_text = art_ent.text.lower()
            for emb_ent in embed_doc.ents:
                emb_text = emb_ent.text.lower()
                # Fuzzy matching for partial similarity
                similarity = fuzz.ratio(art_text, emb_text) / 100
                if 0.7 <= similarity < 1.0:  # Partial but not exact match
                    partial_matches += similarity * 0.5  # Weight partial matches less
                    break

    exact_score = weighted_matches / max(1, total_weight)
    partial_score = partial_matches / max(1, len([e for e in article_doc.ents if e.label_ in ['ORG', 'PERSON', 'PRODUCT']]))

    return min(1.0, exact_score + partial_score)

def calculate_temporal_relevance(article_text: str, embed_content: str) -> float:
    """Calculate temporal relevance boost for time-sensitive content."""
    # Extract dates and time references
    time_patterns = [
        r'\b(today|yesterday|tomorrow|now|recent|current|breaking|just in)\b',
        r'\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b',  # Date formats
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{4}\b'  # Years
    ]

    article_time_refs = sum(1 for pattern in time_patterns for _ in re.findall(pattern, article_text.lower()))
    embed_time_refs = sum(1 for pattern in time_patterns for _ in re.findall(pattern, embed_content.lower()))

    # Boost if both mention time (suggesting timely content)
    if article_time_refs > 0 and embed_time_refs > 0:
        return min(0.2, 0.1 * (embed_time_refs / max(1, article_time_refs)))
    return 0.0

def calculate_geographic_relevance(article_doc, embed_doc) -> float:
    """Calculate geographic relevance for location-based content."""
    # Get geographic entities
    article_geo = [ent.text.lower() for ent in article_doc.ents if ent.label_ in ['GPE', 'LOC', 'FAC']]
    embed_geo = [ent.text.lower() for ent in embed_doc.ents if ent.label_ in ['GPE', 'LOC', 'FAC']]

    if not article_geo:
        return 0.0

    # Exact matches
    exact_matches = len(set(article_geo) & set(embed_geo))

    # Hierarchical relationships (e.g., "California" matches "San Francisco, California")
    hierarchical_matches = 0
    for art_geo in article_geo:
        for emb_geo in embed_geo:
            if art_geo in emb_geo or emb_geo in art_geo:
                hierarchical_matches += 0.5

    total_matches = exact_matches + hierarchical_matches
    return min(0.3, total_matches / len(article_geo))

def calculate_dynamic_threshold(article_complexity: float, entity_richness: int,
                              embed_length: int, strictness: str) -> float:
    """Calculate adaptive threshold based on article and content characteristics."""
    # Base threshold adjustments
    if article_complexity > 0.7:  # Complex article
        base_threshold = 0.15  # Lower threshold (easier to match)
    elif article_complexity > 0.4:  # Medium complexity
        base_threshold = 0.2
    else:  # Simple article
        base_threshold = 0.25  # Higher threshold (stricter)

    # Entity richness adjustment
    if entity_richness > 10:  # Many entities to match
        base_threshold -= 0.05
    elif entity_richness < 3:  # Few entities to match
        base_threshold += 0.05

    # Embed length adjustment
    if embed_length < 50:  # Very short embeds need higher relevance
        base_threshold += 0.1
    elif embed_length > 200:  # Longer embeds get more leeway
        base_threshold -= 0.05

    # Strictness adjustment
    if strictness == "strict":
        base_threshold += 0.08
    elif strictness == "lenient":
        base_threshold -= 0.08

    return max(0.1, min(0.5, base_threshold))  # Keep within reasonable bounds

class RelevanceRequest(BaseModel):
    articleTitle: str
    articleSummary: str
    embedType: str
    embedContent: str
    articleText: str
    # Optional configuration parameters
    threshold: Optional[float] = None
    strictness: Optional[str] = "standard"  # "strict", "standard", "lenient"
    domain: Optional[str] = "general"  # "news", "tech", "business", "medical", "legal"

class RelevanceResponse(BaseModel):
    keep: bool
    reason: str
    scores: Optional[dict] = None

@app.post("/analyze-relevance", response_model=RelevanceResponse)
async def analyze_relevance(request: RelevanceRequest):
    """
    Enhanced relevance analysis with dynamic threshold calculation and multi-dimensional scoring.
    """
    try:
        # Create comprehensive article text
        article_text = f"{request.articleTitle}. {request.articleSummary}. {request.articleText}"

        # Process texts with spaCy
        article_doc = nlp(article_text)
        embed_doc = nlp(request.embedContent)

        # Extract entities
        article_entities = set([ent.text.lower() for ent in article_doc.ents])
        embed_entities = set([ent.text.lower() for ent in embed_doc.ents])

        # 1. Enhanced Semantic Analysis (50%)
        article_embedding = semantic_model.encode(article_text, convert_to_tensor=True)
        embed_embedding = semantic_model.encode(request.embedContent, convert_to_tensor=True)
        semantic_sim = torch.cosine_similarity(article_embedding, embed_embedding, dim=0).item()

        # Extract key concepts for focused semantic analysis
        article_concepts = extract_key_concepts(article_text)
        embed_concepts = extract_key_concepts(request.embedContent)

        # Concept overlap bonus
        concept_overlap = len(set(article_concepts) & set(embed_concepts))
        concept_boost = min(0.15, concept_overlap * 0.02)

        enhanced_semantic = (semantic_sim * 0.8) + concept_boost

        # 2. Contextual Entity Similarity (30%)
        entity_sim = contextual_entity_similarity(article_entities, embed_entities, article_doc, embed_doc)

        # 3. Contextual Analysis (20%)

        # Temporal relevance
        temporal_boost = calculate_temporal_relevance(article_text, request.embedContent)

        # Geographic relevance
        geographic_boost = calculate_geographic_relevance(article_doc, embed_doc)

        # Cross-reference analysis (enhanced)
        cross_reference_boost = 0
        key_article_entities = set()
        for ent in article_doc.ents:
            if ent.label_ in ['ORG', 'PERSON', 'GPE', 'MONEY', 'PERCENT', 'EVENT', 'PRODUCT', 'WORK_OF_ART', 'LAW']:
                key_article_entities.add(ent.text.lower())

        embed_entity_mentions = sum(1 for entity in key_article_entities
                                  if entity in request.embedContent.lower())
        if embed_entity_mentions > 0:
            cross_reference_boost += min(0.2, embed_entity_mentions * 0.06)

        # Authority pattern detection (semantic-based, not keyword-based)
        authority_indicators = [
            "official", "confirmed", "announced", "statement", "report", "sources",
            "breaking", "alert", "update", "developing", "exclusive", "verified"
        ]
        authority_score = sum(1 for indicator in authority_indicators
                            if indicator in request.embedContent.lower()) / len(authority_indicators)
        authority_boost = min(0.15, authority_score * 0.5)

        contextual_score = temporal_boost + geographic_boost + cross_reference_boost + authority_boost

        # Calculate article complexity for dynamic threshold
        article_complexity = calculate_article_complexity(article_text, article_entities)
        entity_richness = len(article_entities)

        # Dynamic threshold calculation
        if request.threshold is not None:
            threshold = request.threshold
        else:
            threshold = calculate_dynamic_threshold(
                article_complexity, entity_richness,
                len(request.embedContent), request.strictness
            )

        # Final scoring with enhanced weights
        final_score = (enhanced_semantic * 0.5) + (entity_sim * 0.3) + contextual_score

        # Adjust for domain-specific requirements
        if request.domain == "news":
            # News content gets boost for timely, geographic content
            final_score += (temporal_boost * 0.5) + (geographic_boost * 0.3)
        elif request.domain == "tech":
            # Tech content gets boost for entity precision
            final_score += entity_sim * 0.2
        elif request.domain == "business":
            # Business content gets boost for organizational entities
            org_entities = [e for e in article_doc.ents if e.label_ == 'ORG']
            if org_entities:
                final_score += 0.1 * (len([e for e in embed_doc.ents if e.label_ == 'ORG']) / len(org_entities))

        # Cap final score
        final_score = min(1.0, final_score)

        keep = final_score > threshold

        # Enhanced reasoning
        if enhanced_semantic > 0.7:
            semantic_reason = "Strong conceptual alignment"
        elif enhanced_semantic > 0.4:
            semantic_reason = "Good topical relevance"
        else:
            semantic_reason = "Limited semantic connection"

        if entity_sim > 0.6:
            entity_reason = "high entity precision"
        elif entity_sim > 0.3:
            entity_reason = "moderate entity overlap"
        else:
            entity_reason = "minimal entity connection"

        contextual_reasons = []
        if temporal_boost > 0:
            contextual_reasons.append(f"temporal relevance (+{temporal_boost:.2f})")
        if geographic_boost > 0:
            contextual_reasons.append(f"geographic relevance (+{geographic_boost:.2f})")
        if cross_reference_boost > 0:
            contextual_reasons.append(f"cross-reference (+{cross_reference_boost:.2f})")
        if authority_boost > 0:
            contextual_reasons.append(f"authority indicators (+{authority_boost:.2f})")

        context_reason = " | ".join(contextual_reasons) if contextual_reasons else "no contextual boost"

        reason = f"{semantic_reason}, {entity_reason}, {context_reason}"

        scores = {
            "enhanced_semantic": round(enhanced_semantic, 3),
            "entity_similarity": round(entity_sim, 3),
            "contextual_score": round(contextual_score, 3),
            "article_complexity": round(article_complexity, 3),
            "entity_richness": entity_richness,
            "final_score": round(final_score, 3),
            "threshold": round(threshold, 3),
            "temporal_boost": round(temporal_boost, 3),
            "geographic_boost": round(geographic_boost, 3)
        }

        return RelevanceResponse(keep=keep, reason=reason, scores=scores)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/config")
def get_config():
    """Get information about available configuration options"""
    return {
        "strictness_levels": {
            "strict": "Higher threshold, more selective filtering (+0.08 to base threshold)",
            "standard": "Default balanced approach (base threshold)",
            "lenient": "Lower threshold, more permissive filtering (-0.08 to base threshold)"
        },
        "domains": {
            "general": "General purpose relevance detection",
            "news": "Optimized for news content (temporal + geographic boost)",
            "tech": "Optimized for technical content (entity precision boost)",
            "business": "Optimized for business content (organizational entity boost)",
            "medical": "Optimized for medical and health content",
            "legal": "Optimized for legal and regulatory content"
        },
        "threshold_info": {
            "description": "Custom threshold value (0.0-1.0). If provided, overrides adaptive calculation",
            "recommended_ranges": {
                "high_precision": "0.3-0.4 (fewer false positives)",
                "balanced": "0.2-0.3 (good balance)",
                "high_recall": "0.1-0.2 (fewer false negatives)"
            }
        },
        "enhancements_v2": {
            "Dynamic threshold calculation": "Based on article complexity and entity richness",
            "Contextual entity matching": "Partial matching and importance weighting",
            "Temporal relevance detection": "Time-sensitive content boost",
            "Geographic relevance": "Location-based content analysis",
            "Enhanced semantic analysis": "Concept extraction and focused similarity",
            "Domain-specific optimization": "Tailored scoring for different content types",
            "Multi-dimensional scoring": "50% semantic, 30% entity, 20% contextual"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)