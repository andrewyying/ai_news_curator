"""Data models for the AI News Curator."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, HttpUrl


class RawNewsItem(BaseModel):
    """Raw news item fetched from RSS feeds."""
    
    id: str  # Internal unique ID, e.g., source + hash(title+url)
    title: str
    url: Optional[HttpUrl] = None
    source: str
    published_at: Optional[datetime] = None
    content: str  # Article body or summary text


class ClassifiedNewsItem(RawNewsItem):
    """News item with classification."""
    
    category: str  # e.g., "AI Models", "AI Infra", ...
    classification_confidence: float
    classification_method: str  # "zero-shot" or "few-shot"


class ScoredNewsItem(ClassifiedNewsItem):
    """News item with impact score."""
    
    impact_score: int  # 1~5
    impact_reason: str  # LLM explanation
    impact_dimensions: List[str]  # e.g., ["industry", "career", "research"]


class ClusteredItem(BaseModel):
    """Cluster of similar news items."""
    
    cluster_id: str
    representative: ScoredNewsItem
    members: List[ScoredNewsItem]


class SummarizedCluster(BaseModel):
    """Final summarized cluster for report generation."""
    
    cluster_id: str
    category: str
    impact_score: int
    title: str  # Aggregated title
    summary: str  # Final summary for daily report
    impact_reason: str
    sources: List[str]  # URL list
    raw_ids: List[str]  # IDs of included RawNewsItem

