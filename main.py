from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import uvicorn
from services.screenshot_service import ScreenshotService
from services.relevance_service import RelevanceService

app = FastAPI(title="Enhanced Embed Relevance Analyzer", version="2.0.0")

# Initialize services
relevance_service = RelevanceService()

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

class ScreenshotRequest(BaseModel):
    url: str
    browserless_url: Optional[str] = None
    browserless_token: Optional[str] = None

class ScreenshotItem(BaseModel):
    type: str
    selector: str
    index: int
    screenshot: str  # Base64 encoded

class ScreenshotResponse(BaseModel):
    url: str
    screenshots: List[ScreenshotItem]
    count: int

@app.post("/analyze-relevance", response_model=RelevanceResponse)
async def analyze_relevance(request: RelevanceRequest):
    """
    Enhanced relevance analysis with dynamic threshold calculation and multi-dimensional scoring.
    """
    try:
        result = relevance_service.analyze(
            title=request.articleTitle,
            summary=request.articleSummary,
            text=request.articleText,
            embed_content=request.embedContent,
            embed_type=request.embedType,
            threshold=request.threshold,
            strictness=request.strictness,
            domain=request.domain
        )
        return RelevanceResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/screenshot-embeds", response_model=ScreenshotResponse)
async def screenshot_embeds(request: ScreenshotRequest):
    """
    Detect and screenshot embeds within a given article URL.
    """
    try:
        service = ScreenshotService(
            browserless_url=request.browserless_url,
            browserless_token=request.browserless_token
        )
        screenshots = await service.get_embed_screenshots(request.url)
        
        return ScreenshotResponse(
            url=request.url,
            screenshots=screenshots,
            count=len(screenshots)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Screenshot process failed: {str(e)}")

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