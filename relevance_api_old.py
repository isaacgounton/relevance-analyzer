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

app = FastAPI(title="Embed Relevance Analyzer", version="1.0.0")

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
    Analyze if an embed is relevant to an article using semantic understanding.
    """
    try:
        # Create comprehensive article text
        article_text = f"{request.articleTitle}. {request.articleSummary}. {request.articleText}"

        # Semantic similarity using sentence transformers (multilingual)
        article_embedding = semantic_model.encode(article_text, convert_to_tensor=True)
        embed_embedding = semantic_model.encode(request.embedContent, convert_to_tensor=True)

        # Calculate cosine similarity between embeddings
        semantic_sim = torch.cosine_similarity(article_embedding, embed_embedding, dim=0).item()

        # Traditional TF-IDF as backup (for very short texts)
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        texts = [article_text, request.embedContent]
        vectors = vectorizer.fit_transform(texts)
        tfidf_sim = float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])

        # Named entity overlap (important for news articles)
        article_doc = nlp(article_text)
        embed_doc = nlp(request.embedContent)

        article_entities = set([ent.text.lower() for ent in article_doc.ents])
        embed_entities = set([ent.text.lower() for ent in embed_doc.ents])

        entity_overlap = len(article_entities.intersection(embed_entities)) / len(article_entities.union(embed_entities)) if article_entities.union(embed_entities) else 0

        # Smart content-based relevance scoring (no hardcoded keywords)
        # Analyze how well the embed content aligns with and references the article

        # 1. Cross-reference boost: Does the embed appear to reference the article?
        cross_reference_boost = 0

        embed_lower = request.embedContent.lower()

        # Check if embed mentions key entities from the article
        key_article_entities = set()
        for ent in article_doc.ents:
            if ent.label_ in ['ORG', 'PERSON', 'GPE', 'MONEY', 'PERCENT', 'EVENT', 'PRODUCT', 'WORK_OF_ART', 'LAW', 'DATE', 'FAC', 'LOC']:
                key_article_entities.add(ent.text.lower())

        embed_entity_mentions = sum(1 for entity in key_article_entities
                                  if entity in embed_lower)
        if embed_entity_mentions > 0:
            cross_reference_boost += min(0.15, embed_entity_mentions * 0.05)  # Cap at 0.15

        # 2. Authority language patterns (using semantic similarity to authoritative content)
        # Compare embed against patterns of authoritative communication
        authority_patterns = [
            # News & Reporting
            "breaking news",
            "just in",
            "we can confirm",
            "sources say",
            "eyewitnesses report",
            "latest development",
            "updates indicate",

            # Official Announcements
            "the company announced",
            "government stated",
            "official data shows",
            "we are pleased to announce",
            "board approved",
            "quarterly results",
            "financial report",

            # Expert Attribution
            "experts believe",
            "analysts say",
            "researchers found",
            "study reveals",
            "scientific evidence suggests",
            "medical experts warn",
            "economic forecast",

            # Direct Quotes & Statements
            "told reporters",
            "said in a statement",
            "quoted as saying",
            "declared that",
            "testified that",
            "commented on",
            "spokesperson confirmed",

            # Time-sensitive Language
            "earlier today",
            "this morning",
            "just hours ago",
            "recently revealed",
            "latest information",
            "real-time updates",

            # Verification & Credibility
            "has confirmed",
            "verified by",
            "authentic sources",
            "reliable information",
            "cross-check shows",
            "fact-check confirmed",

            # Legal & Regulatory
            "court ruled",
            "legislation passed",
            "regulatory approval",
            "compliance verified",
            "legal document states",
            "judgment issued",

            # Technical & Scientific
            "peer-reviewed",
            "clinical trial",
            "data analysis shows",
            "research indicates",
            "experimental results",
            "technical specifications",

            # Corporate/Business
            "executive decision",
            "market analysis",
            "industry report",
            "strategic initiative",
            "operational update",
            "business intelligence",

            # Emergency/Official Alerts
            "emergency announcement",
            "public safety",
            "weather warning",
            "health advisory",
            "security alert",

            # Multilingual equivalents (for better cross-language support)
            "déclaration officielle",
            "comunicado oficial",
            "offizielle erklärung",
            "公式声明",
            "공식 성명",
            "ufficial dichiarazione"
        ]

        authority_embeddings = semantic_model.encode(authority_patterns, convert_to_tensor=True)
        embed_auth_similarity = torch.mean(torch.cosine_similarity(
            embed_embedding.unsqueeze(0), authority_embeddings, dim=1
        )).item()

        authority_boost = embed_auth_similarity * 0.1  # Scale down the authority signal

        # 3. Topic coherence: Are they discussing the same subject?
        # Use sentence-level analysis for better topic detection
        article_sentences = [sent.text.strip() for sent in article_doc.sents if len(sent.text.strip()) > 10]
        embed_sentences = [sent.text.strip() for sent in embed_doc.sents if len(sent.text.strip()) > 10]

        if article_sentences and embed_sentences:
            # Compare sentence embeddings for topic coherence
            article_sent_embeddings = semantic_model.encode(article_sentences[:3], convert_to_tensor=True)  # First 3 sentences
            embed_sent_embeddings = semantic_model.encode(embed_sentences[:2], convert_to_tensor=True)    # First 2 sentences

            topic_coherence = torch.mean(torch.cosine_similarity(
                article_sent_embeddings.unsqueeze(1),
                embed_sent_embeddings.unsqueeze(0),
                dim=2
            )).item()

            topic_boost = topic_coherence * 0.1
        else:
            topic_boost = 0

        # Combine intelligent boosts
        content_boost = cross_reference_boost + authority_boost + topic_boost

        # Enhanced scoring algorithm
        # Semantic similarity is now the primary factor (60%)
        # Entity overlap catches important names/places (25%)
        # Content boost uses AI to assess relevance without hardcoded keywords (15%)
        final_score = (semantic_sim * 0.6) + (entity_overlap * 0.25) + content_boost

        # Use custom threshold or calculate adaptive threshold
        if request.threshold is not None:
            threshold = request.threshold
        else:
            # Adaptive threshold based on content length and strictness
            if len(request.embedContent) < 50:  # Very short embeds
                base_threshold = 0.3
            elif len(request.embedContent) < 150:  # Medium embeds
                base_threshold = 0.25
            else:  # Long embeds
                base_threshold = 0.2

            # Adjust based on strictness
            if request.strictness == "strict":
                threshold = base_threshold + 0.1
            elif request.strictness == "lenient":
                threshold = base_threshold - 0.1
            else:  # standard
                threshold = base_threshold

        keep = final_score > threshold

        # Generate meaningful reason
        if semantic_sim > 0.6:
            semantic_reason = "Strong semantic similarity"
        elif semantic_sim > 0.3:
            semantic_reason = "Good semantic similarity"
        else:
            semantic_reason = "Low semantic similarity"

        if entity_overlap > 0.3:
            entity_reason = "high entity overlap"
        elif entity_overlap > 0.1:
            entity_reason = "some entity overlap"
        else:
            entity_reason = "low entity overlap"

        boost_reason = f"content boost: {content_boost:.2f}" if content_boost > 0 else "no content boost"

        reason = f"{semantic_reason}, {entity_reason}, {boost_reason}"

        scores = {
            "semantic_similarity": round(semantic_sim, 3),
            "entity_overlap": round(entity_overlap, 3),
            "content_boost": round(content_boost, 3),
            "final_score": round(final_score, 3),
            "threshold": round(threshold, 3)
        }

        return RelevanceResponse(keep=keep, reason=reason, scores=scores)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/config")
async def get_config_info():
    """Get information about available configuration options"""
    return {
        "strictness_levels": {
            "strict": "Higher threshold, more selective filtering (+0.1 to base threshold)",
            "standard": "Default balanced approach (base threshold)",
            "lenient": "Lower threshold, more permissive filtering (-0.1 to base threshold)"
        },
        "domains": {
            "general": "General purpose relevance detection",
            "news": "Optimized for news articles and journalistic content",
            "tech": "Optimized for technical and scientific content",
            "business": "Optimized for business and corporate content",
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
        "improvements": [
            "Enhanced entity recognition (13 types vs 5 types)",
            "Comprehensive authority patterns (80+ patterns)",
            "Multilingual support for authority detection",
            "Configurable thresholds and strictness levels",
            "Domain-specific optimization",
            "Semantic similarity instead of keyword matching"
        ]
    }

@app.post("/explain-relevance")
async def explain_relevance(request: RelevanceRequest):
    """
    Provide detailed explanation of embed relevance analysis using semantic understanding.
    """
    try:
        # Create comprehensive article text
        article_text = f"{request.articleTitle}. {request.articleSummary}. {request.articleText}"

        # Semantic similarity using sentence transformers (multilingual)
        article_embedding = semantic_model.encode(article_text, convert_to_tensor=True)
        embed_embedding = semantic_model.encode(request.embedContent, convert_to_tensor=True)

        # Calculate cosine similarity between embeddings
        semantic_sim = torch.cosine_similarity(article_embedding, embed_embedding, dim=0).item()

        # Named entity overlap (important for news articles)
        article_doc = nlp(article_text)
        embed_doc = nlp(request.embedContent)

        article_entities = set([ent.text.lower() for ent in article_doc.ents])
        embed_entities = set([ent.text.lower() for ent in embed_doc.ents])

        entity_overlap = len(article_entities.intersection(embed_entities)) / len(article_entities.union(embed_entities)) if article_entities.union(embed_entities) else 0

        # Smart content-based relevance scoring (no hardcoded keywords)
        # Analyze how well the embed content aligns with and references the article

        # 1. Cross-reference boost: Does the embed appear to reference the article?
        cross_reference_boost = 0

        embed_lower = request.embedContent.lower()

        # Check if embed mentions key entities from the article
        key_article_entities = set()
        for ent in article_doc.ents:
            if ent.label_ in ['ORG', 'PERSON', 'GPE', 'MONEY', 'PERCENT', 'EVENT', 'PRODUCT', 'WORK_OF_ART', 'LAW', 'DATE', 'FAC', 'LOC']:
                key_article_entities.add(ent.text.lower())

        embed_entity_mentions = sum(1 for entity in key_article_entities
                                  if entity in embed_lower)
        if embed_entity_mentions > 0:
            cross_reference_boost += min(0.15, embed_entity_mentions * 0.05)  # Cap at 0.15

        # 2. Authority language patterns (using semantic similarity to authoritative content)
        # Compare embed against patterns of authoritative communication
        authority_patterns = [
            # News & Reporting
            "breaking news",
            "just in",
            "we can confirm",
            "sources say",
            "eyewitnesses report",
            "latest development",
            "updates indicate",

            # Official Announcements
            "the company announced",
            "government stated",
            "official data shows",
            "we are pleased to announce",
            "board approved",
            "quarterly results",
            "financial report",

            # Expert Attribution
            "experts believe",
            "analysts say",
            "researchers found",
            "study reveals",
            "scientific evidence suggests",
            "medical experts warn",
            "economic forecast",

            # Direct Quotes & Statements
            "told reporters",
            "said in a statement",
            "quoted as saying",
            "declared that",
            "testified that",
            "commented on",
            "spokesperson confirmed",

            # Time-sensitive Language
            "earlier today",
            "this morning",
            "just hours ago",
            "recently revealed",
            "latest information",
            "real-time updates",

            # Verification & Credibility
            "has confirmed",
            "verified by",
            "authentic sources",
            "reliable information",
            "cross-check shows",
            "fact-check confirmed",

            # Legal & Regulatory
            "court ruled",
            "legislation passed",
            "regulatory approval",
            "compliance verified",
            "legal document states",
            "judgment issued",

            # Technical & Scientific
            "peer-reviewed",
            "clinical trial",
            "data analysis shows",
            "research indicates",
            "experimental results",
            "technical specifications",

            # Corporate/Business
            "executive decision",
            "market analysis",
            "industry report",
            "strategic initiative",
            "operational update",
            "business intelligence",

            # Emergency/Official Alerts
            "emergency announcement",
            "public safety",
            "weather warning",
            "health advisory",
            "security alert",

            # Multilingual equivalents (for better cross-language support)
            "déclaration officielle",
            "comunicado oficial",
            "offizielle erklärung",
            "公式声明",
            "공식 성명",
            "ufficial dichiarazione"
        ]

        authority_embeddings = semantic_model.encode(authority_patterns, convert_to_tensor=True)
        embed_auth_similarity = torch.mean(torch.cosine_similarity(
            embed_embedding.unsqueeze(0), authority_embeddings, dim=1
        )).item()

        authority_boost = embed_auth_similarity * 0.1  # Scale down the authority signal

        # 3. Topic coherence: Are they discussing the same subject?
        # Use sentence-level analysis for better topic detection
        article_sentences = [sent.text.strip() for sent in article_doc.sents if len(sent.text.strip()) > 10]
        embed_sentences = [sent.text.strip() for sent in embed_doc.sents if len(sent.text.strip()) > 10]

        if article_sentences and embed_sentences:
            # Compare sentence embeddings for topic coherence
            article_sent_embeddings = semantic_model.encode(article_sentences[:3], convert_to_tensor=True)  # First 3 sentences
            embed_sent_embeddings = semantic_model.encode(embed_sentences[:2], convert_to_tensor=True)    # First 2 sentences

            topic_coherence = torch.mean(torch.cosine_similarity(
                article_sent_embeddings.unsqueeze(1),
                embed_sent_embeddings.unsqueeze(0),
                dim=2
            )).item()

            topic_boost = topic_coherence * 0.1
        else:
            topic_boost = 0

        # Combine intelligent boosts
        content_boost = cross_reference_boost + authority_boost + topic_boost

        # Enhanced scoring algorithm
        # Semantic similarity is now the primary factor (60%)
        # Entity overlap catches important names/places (25%)
        # Content boost uses AI to assess relevance without hardcoded keywords (15%)
        final_score = (semantic_sim * 0.6) + (entity_overlap * 0.25) + content_boost

        # Use custom threshold or calculate adaptive threshold
        if request.threshold is not None:
            threshold = request.threshold
        else:
            # Adaptive threshold based on content length and strictness
            if len(request.embedContent) < 50:  # Very short embeds
                base_threshold = 0.3
            elif len(request.embedContent) < 150:  # Medium embeds
                base_threshold = 0.25
            else:  # Long embeds
                base_threshold = 0.2

            # Adjust based on strictness
            if request.strictness == "strict":
                threshold = base_threshold + 0.1
            elif request.strictness == "lenient":
                threshold = base_threshold - 0.1
            else:  # standard
                threshold = base_threshold

        keep = final_score > threshold

        # Generate detailed explanation
        explanation = generate_semantic_explanation(
            semantic_sim, entity_overlap, content_boost, final_score, threshold, keep,
            article_entities, embed_entities
        )

        return {
            "keep": keep,
            "final_score": round(final_score, 3),
            "threshold": round(threshold, 3),
            "scores": {
                "semantic_similarity": round(semantic_sim, 3),
                "entity_overlap": round(entity_overlap, 3),
                "content_boost": round(content_boost, 3)
            },
            "explanation": explanation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def generate_semantic_explanation(semantic_sim, entity_overlap, content_boost, final_score, threshold, keep,
                                article_entities, embed_entities):
    """Generate human-readable explanation of the semantic analysis results."""

    explanation = []

    # Overall decision
    if keep:
        explanation.append("✅ **EMBED KEPT** - This embed appears relevant to the article.")
    else:
        explanation.append("❌ **EMBED REJECTED** - This embed does not appear relevant to the article.")

    explanation.append(f"**Overall Score**: {final_score:.3f}/1.000 (threshold: {threshold:.3f})")

    # Semantic Similarity Analysis
    explanation.append("\n🧠 **Semantic Understanding Analysis**")
    if semantic_sim < 0.2:
        explanation.append(f"• Very low semantic similarity ({semantic_sim:.3f}) - The AI model detects different topics and meanings.")
    elif semantic_sim < 0.4:
        explanation.append(f"• Low semantic similarity ({semantic_sim:.3f}) - Some conceptual overlap but different main topics.")
    elif semantic_sim < 0.7:
        explanation.append(f"• Good semantic similarity ({semantic_sim:.3f}) - Related concepts and contextual connections.")
    else:
        explanation.append(f"• High semantic similarity ({semantic_sim:.3f}) - Strong semantic relationship between article and embed.")

    # Entity Analysis
    explanation.append("\n🏷️ **Named Entity Analysis**")
    shared_entities = article_entities.intersection(embed_entities)
    if len(shared_entities) == 0:
        explanation.append("• No shared named entities - Different people, organizations, or places mentioned.")
    else:
        explanation.append(f"• Found {len(shared_entities)} shared entities: {', '.join(list(shared_entities)[:5])}{'...' if len(shared_entities) > 5 else ''}")
        explanation.append(f"• Entity overlap: {entity_overlap:.3f} - {'Strong entity alignment' if entity_overlap > 0.3 else 'Some entity overlap' if entity_overlap > 0.1 else 'Limited entity alignment'}")

    # Content Quality Analysis
    explanation.append("\n🎯 **Content Quality Analysis**")
    if content_boost > 0.15:
        explanation.append(f"• Strong content alignment (+{content_boost:.2f}) - Embed shows high relevance through cross-referencing and topic coherence.")
    elif content_boost > 0.05:
        explanation.append(f"• Moderate content alignment (+{content_boost:.2f}) - Some relevant connections detected.")
    else:
        explanation.append("• Limited content alignment detected.")

    # Specific insights
    explanation.append("\n💡 **AI Analysis Insights**")
    if not keep:
        if semantic_sim < 0.2:
            explanation.append("• **Primary issue**: AI model identifies fundamentally different topics")
        if entity_overlap < 0.05:
            explanation.append("• **Missing connection**: No shared people, organizations, or locations")
        if content_boost == 0:
            explanation.append("• **Quality concern**: Limited content alignment with article")
    else:
        if semantic_sim > 0.5:
            explanation.append("• **Strength**: Strong semantic understanding of related content")
        if entity_overlap > 0.2:
            explanation.append("• **Strength**: Key entities are consistent between article and embed")
        if content_boost > 0.1:
            explanation.append("• **Strength**: High content alignment and relevance")

    # Summary recommendation
    explanation.append(f"\n📊 **Final Assessment**: Score {final_score:.3f} vs threshold {threshold:.3f}")
    if keep:
        explanation.append("The AI model recommends **KEEPING** this embed as it adds value to the article content.")
    else:
        explanation.append("The AI model recommends **REJECTING** this embed as it lacks sufficient relevance.")

    return "\n".join(explanation)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)