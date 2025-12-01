"""Summary generation module with concurrent processing."""

import asyncio
import json
from typing import List
from tqdm.asyncio import tqdm as async_tqdm

from models import ClusteredItem, SummarizedCluster
from llm import call_llm_json, load_prompt


async def _summarize_single_cluster(
    cluster: ClusteredItem,
    prompt_template: str,
    semaphore: asyncio.Semaphore,
) -> SummarizedCluster:
    """Summarize a single cluster asynchronously."""
    async with semaphore:
        try:
            # Prepare articles list for LLM (reduced content length for efficiency)
            articles = []
            for member in cluster.members:
                articles.append({
                    "title": member.title,
                    "content": member.content[:500],  # Reduced from 1000 to 500 for faster processing
                    "source": member.source,
                    "impact_score": member.impact_score,
                    "impact_reason": member.impact_reason,
                })
            
            input_json = json.dumps({"articles": articles}, ensure_ascii=False)
            
            # Construct prompt
            prompt = f"{prompt_template}\n\nArticles:\n{input_json}"
            
            # Call LLM (run in thread pool to avoid blocking)
            response = await asyncio.to_thread(call_llm_json, prompt, temperature=0.3)
            
            title = response.get("title", cluster.representative.title)
            summary = response.get("summary", "")
            responsible_ai_notes = response.get("responsible_ai_notes", "")
            
            # Collect sources and raw IDs
            sources = []
            raw_ids = []
            for member in cluster.members:
                if member.url:
                    sources.append(str(member.url))
                raw_ids.append(member.id)
            
            # Combine impact_reason with responsible_ai_notes
            impact_reason = cluster.representative.impact_reason
            if responsible_ai_notes:
                impact_reason += f"\n\nResponsible AI Notes: {responsible_ai_notes}"
            
            # Create summarized cluster
            return SummarizedCluster(
                cluster_id=cluster.cluster_id,
                category=cluster.representative.category,
                impact_score=cluster.representative.impact_score,
                title=title,
                summary=summary,
                impact_reason=impact_reason,
                sources=list(set(sources)),  # Remove duplicates
                raw_ids=raw_ids,
            )
        
        except Exception as e:
            print(f"Error summarizing cluster {cluster.cluster_id}: {e}")
            # Fallback to basic summary
            sources = []
            raw_ids = []
            for member in cluster.members:
                if member.url:
                    sources.append(str(member.url))
                raw_ids.append(member.id)
            
            return SummarizedCluster(
                cluster_id=cluster.cluster_id,
                category=cluster.representative.category,
                impact_score=cluster.representative.impact_score,
                title=cluster.representative.title,
                summary=f"Error generating summary: {str(e)}",
                impact_reason=cluster.representative.impact_reason,
                sources=list(set(sources)),
                raw_ids=raw_ids,
            )


async def summarize_clusters_async(
    clusters: List[ClusteredItem],
    max_concurrent: int = 10,
) -> List[SummarizedCluster]:
    """
    Generate summaries for clustered news items with concurrent processing.
    
    Args:
        clusters: List of clustered items
        max_concurrent: Maximum number of concurrent API calls
        
    Returns:
        List of summarized clusters
    """
    prompt_template = load_prompt("summary_prompt")
    
    print(f"Summarizing {len(clusters)} clusters (concurrent={max_concurrent})...")
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Process all clusters concurrently
    tasks = [
        _summarize_single_cluster(cluster, prompt_template, semaphore)
        for cluster in clusters
    ]
    results = await async_tqdm.gather(*tasks, desc="Summarizing")
    
    return list(results)


def summarize_clusters(
    clusters: List[ClusteredItem],
    max_concurrent: int = 10,
) -> List[SummarizedCluster]:
    """
    Generate summaries for clustered news items (synchronous wrapper).
    
    Args:
        clusters: List of clustered items
        max_concurrent: Maximum number of concurrent API calls
        
    Returns:
        List of summarized clusters
    """
    return asyncio.run(summarize_clusters_async(clusters, max_concurrent))

