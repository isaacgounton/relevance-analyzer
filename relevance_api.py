from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from thefuzz import fuzz
import uvicorn
from typing import Optional
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

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

class RelevanceRequest(BaseModel):
    articleTitle: str
    articleSummary: str
    embedType: str
    embedContent: str
    articleText: str

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

        # Check for exact source mentions (official accounts, organizations)
        source_mention_boost = 0
        article_lower = article_text.lower()
        embed_lower = request.embedContent.lower()

        # Boost for official sources
        if any(official in embed_lower for official in [
            "banque du canada", "bank of canada", "official", "officiel",
            "gouvernement", "government", "ministère", "ministry"
        ]):
            source_mention_boost += 0.2

        # Boost for direct topic mentions
        if any(keyword in embed_lower for keyword in [
            "taux directeur", "interest rate", "taux", "rate", "décision", "decision"
        ]):
            source_mention_boost += 0.15

        # Enhanced scoring algorithm
        # Semantic similarity is now the primary factor (60%)
        # Entity overlap catches important names/places (25%)
        # Source mention boost rewards official/relevant content (15%)
        final_score = (semantic_sim * 0.6) + (entity_overlap * 0.25) + source_mention_boost

        # Adaptive threshold based on content length
        if len(request.embedContent) < 50:  # Very short embeds
            threshold = 0.3
        elif len(request.embedContent) < 150:  # Medium embeds
            threshold = 0.25
        else:  # Long embeds
            threshold = 0.2

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

        boost_reason = f"source boost: {source_mention_boost:.2f}" if source_mention_boost > 0 else "no source boost"

        reason = f"{semantic_reason}, {entity_reason}, {boost_reason}"

        scores = {
            "semantic_similarity": round(semantic_sim, 3),
            "entity_overlap": round(entity_overlap, 3),
            "source_boost": round(source_mention_boost, 3),
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

        # Check for exact source mentions (official accounts, organizations)
        source_mention_boost = 0
        article_lower = article_text.lower()
        embed_lower = request.embedContent.lower()

        # Boost for official sources
        if any(official in embed_lower for official in [
            "banque du canada", "bank of canada", "official", "officiel",
            "gouvernement", "government", "ministère", "ministry"
        ]):
            source_mention_boost += 0.2

        # Boost for direct topic mentions
        if any(keyword in embed_lower for keyword in [
            "taux directeur", "interest rate", "taux", "rate", "décision", "decision"
        ]):
            source_mention_boost += 0.15

        # Enhanced scoring algorithm
        final_score = (semantic_sim * 0.6) + (entity_overlap * 0.25) + source_mention_boost

        # Adaptive threshold based on content length
        if len(request.embedContent) < 50:  # Very short embeds
            threshold = 0.3
        elif len(request.embedContent) < 150:  # Medium embeds
            threshold = 0.25
        else:  # Long embeds
            threshold = 0.2

        keep = final_score > threshold

        # Generate detailed explanation
        explanation = generate_semantic_explanation(
            semantic_sim, entity_overlap, source_mention_boost, final_score, threshold, keep,
            article_entities, embed_entities, request
        )

        return {
            "keep": keep,
            "final_score": round(final_score, 3),
            "threshold": round(threshold, 3),
            "scores": {
                "semantic_similarity": round(semantic_sim, 3),
                "entity_overlap": round(entity_overlap, 3),
                "source_boost": round(source_mention_boost, 3)
            },
            "explanation": explanation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def generate_semantic_explanation(semantic_sim, entity_overlap, source_boost, final_score, threshold, keep,
                                article_entities, embed_entities, request):
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

    # Source Quality Analysis
    explanation.append("\n🏛️ **Source Quality Analysis**")
    if source_boost > 0.15:
        explanation.append(f"• Official source detected (+{source_boost:.2f}) - Embed appears to be from an authoritative source.")
    elif source_boost > 0.05:
        explanation.append(f"• Topic-relevant content (+{source_boost:.2f}) - Embed discusses relevant subject matter.")
    else:
        explanation.append("• No source quality boost detected.")

    # Specific insights
    explanation.append("\n💡 **AI Analysis Insights**")
    if not keep:
        if semantic_sim < 0.2:
            explanation.append("• **Primary issue**: AI model identifies fundamentally different topics")
        if entity_overlap < 0.05:
            explanation.append("• **Missing connection**: No shared people, organizations, or locations")
        if source_boost == 0:
            explanation.append("• **Quality concern**: No evidence of official or authoritative source")
    else:
        if semantic_sim > 0.5:
            explanation.append("• **Strength**: Strong semantic understanding of related content")
        if entity_overlap > 0.2:
            explanation.append("• **Strength**: Key entities are consistent between article and embed")
        if source_boost > 0.1:
            explanation.append("• **Strength**: High-quality source alignment")

    # Summary recommendation
    explanation.append(f"\n📊 **Final Assessment**: Score {final_score:.3f} vs threshold {threshold:.3f}")
    if keep:
        explanation.append("The AI model recommends **KEEPING** this embed as it adds value to the article content.")
    else:
        explanation.append("The AI model recommends **REJECTING** this embed as it lacks sufficient relevance.")

    return "\n".join(explanation)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)