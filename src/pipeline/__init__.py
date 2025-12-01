"""Pipeline modules for news processing."""

from .classify import classify_zero_shot, classify_few_shot, CATEGORIES
from .impact import score_impact
from .deduplicate import cluster_items
from .summarize import summarize_clusters
from .report import generate_markdown_report

__all__ = [
    "classify_zero_shot",
    "classify_few_shot",
    "CATEGORIES",
    "score_impact",
    "cluster_items",
    "summarize_clusters",
    "generate_markdown_report",
]

