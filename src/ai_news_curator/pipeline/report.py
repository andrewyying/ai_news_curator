"""Markdown report generation module."""

from datetime import date
from typing import List
from collections import defaultdict

from ..models import SummarizedCluster


def generate_markdown_report(
    clusters: List[SummarizedCluster],
    report_date: date,
) -> str:
    """
    Generate markdown report from summarized clusters.
    
    Args:
        clusters: List of summarized clusters
        report_date: Date for the report
        
    Returns:
        Markdown string
    """
    # Sort by impact score (descending)
    sorted_clusters = sorted(clusters, key=lambda x: x.impact_score, reverse=True)
    
    # Group by category
    by_category = defaultdict(list)
    for cluster in sorted_clusters:
        by_category[cluster.category].append(cluster)
    
    # Extract high-impact items
    impact_5 = [c for c in sorted_clusters if c.impact_score == 5]
    impact_4 = [c for c in sorted_clusters if c.impact_score == 4]
    
    # Count merged items
    merged_items = [c for c in sorted_clusters if len(c.sources) > 1]
    
    # Collect responsible AI notes
    responsible_ai_notes = []
    for cluster in sorted_clusters:
        if "Responsible AI Notes:" in cluster.impact_reason:
            note = cluster.impact_reason.split("Responsible AI Notes:")[-1].strip()
            if note:
                responsible_ai_notes.append((cluster.title, note))
    
    # Build markdown
    lines = []
    lines.append(f"# AI News Curator Daily Report")
    lines.append(f"**Date:** {report_date.isoformat()}")
    lines.append(f"**Total Stories:** {len(clusters)}")
    lines.append("")
    
    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    top_items = sorted_clusters[:min(5, len(sorted_clusters))]
    for item in top_items:
        lines.append(f"- **{item.title}** ({item.category}, Impact: {item.impact_score})")
        lines.append(f"  {item.summary[:200]}...")
        lines.append("")
    
    # Most Important (Impact 5)
    if impact_5:
        lines.append("## Most Important (Impact Score: 5)")
        lines.append("")
        for item in impact_5:
            lines.append(f"### {item.title}")
            lines.append(f"**Category:** {item.category}")
            lines.append(f"**Impact Score:** {item.impact_score}")
            lines.append("")
            lines.append(f"**Summary:**")
            lines.append(item.summary)
            lines.append("")
            lines.append(f"**Why it matters:**")
            lines.append(item.impact_reason.split("Responsible AI Notes:")[0].strip())
            lines.append("")
            if item.sources:
                lines.append("**Sources:**")
                for source in item.sources[:3]:  # Limit to 3 sources
                    lines.append(f"- {source}")
                if len(item.sources) > 3:
                    lines.append(f"- ... and {len(item.sources) - 3} more")
                lines.append("")
    
    # High Priority (Impact 4)
    if impact_4:
        lines.append("## High Priority (Impact Score: 4)")
        lines.append("")
        for item in impact_4:
            lines.append(f"### {item.title}")
            lines.append(f"**Category:** {item.category}")
            lines.append("")
            lines.append(item.summary)
            lines.append("")
            if item.sources:
                lines.append("**Sources:**")
                for source in item.sources[:2]:
                    lines.append(f"- {source}")
                if len(item.sources) > 2:
                    lines.append(f"- ... and {len(item.sources) - 2} more")
                lines.append("")
    
    # By Category
    lines.append("## News by Category")
    lines.append("")
    for category in sorted(by_category.keys()):
        category_items = by_category[category]
        lines.append(f"### {category} ({len(category_items)} items)")
        lines.append("")
        
        for item in category_items:
            lines.append(f"#### {item.title}")
            lines.append(f"*Impact Score: {item.impact_score}*")
            lines.append("")
            lines.append(item.summary)
            lines.append("")
            if item.sources:
                lines.append("**Sources:**")
                for source in item.sources[:2]:
                    lines.append(f"- {source}")
                if len(item.sources) > 2:
                    lines.append(f"- ... and {len(item.sources) - 2} more")
                lines.append("")
    
    # Merged Items
    if merged_items:
        lines.append("## Merged / Duplicate Stories")
        lines.append("")
        lines.append(f"The following {len(merged_items)} stories were merged from multiple sources:")
        lines.append("")
        for item in merged_items:
            lines.append(f"- **{item.title}**: Merged from {len(item.sources)} sources")
            lines.append(f"  - {', '.join(item.sources[:3])}")
            if len(item.sources) > 3:
                lines.append(f"  - ... and {len(item.sources) - 3} more")
            lines.append("")
    
    # Responsible AI Notes
    if responsible_ai_notes:
        lines.append("## Responsible AI Notes")
        lines.append("")
        lines.append("The following items include notes on potential concerns:")
        lines.append("")
        for title, note in responsible_ai_notes:
            lines.append(f"### {title}")
            lines.append(note)
            lines.append("")
    
    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Generated by AI News Curator*")
    
    return "\n".join(lines)

