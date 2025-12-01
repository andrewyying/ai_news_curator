"""Impact scoring module with concurrent processing and caching."""

import asyncio
import json
from datetime import date
from typing import List, Optional
from tqdm.asyncio import tqdm as async_tqdm

from ..models import ClassifiedNewsItem, ScoredNewsItem
from ..llm import call_llm_json, load_prompt
from ..cache import NewsCache


async def _score_single_item(
    item: ClassifiedNewsItem,
    prompt_template: str,
    cache: Optional[NewsCache],
    target_date: date,
    semaphore: asyncio.Semaphore,
) -> ScoredNewsItem:
    """Score a single news item asynchronously."""
    async with semaphore:
        # Check cache first
        if cache:
            cached = cache.get_scored(item.id, target_date)
            if cached:
                return cached
        
        try:
            # Prepare input JSON (reduced content length for efficiency)
            input_data = {
                "title": item.title,
                "content": item.content[:800],  # Reduced from 2000 to 800 for faster processing
                "category": item.category,
            }
            input_json = json.dumps(input_data, ensure_ascii=False)
            
            # Construct prompt
            prompt = f"{prompt_template}\n\nNews item:\n{input_json}"
            
            # Call LLM (run in thread pool to avoid blocking)
            response = await asyncio.to_thread(call_llm_json, prompt, temperature=0.3)
            
            impact_score = int(response.get("impact_score", 3))
            impact_dimensions = response.get("impact_dimensions", [])
            impact_reason = response.get("impact_reason", "")
            
            # Validate impact score
            impact_score = max(1, min(5, impact_score))
            
            # Ensure impact_dimensions is a list
            if not isinstance(impact_dimensions, list):
                impact_dimensions = []
            
            # Create scored item
            scored_item = ScoredNewsItem(
                **item.model_dump(),
                impact_score=impact_score,
                impact_reason=impact_reason,
                impact_dimensions=impact_dimensions,
            )
            
            # Save to cache
            if cache:
                cache.save_scored(scored_item, target_date)
            
            return scored_item
        
        except Exception as e:
            print(f"Error scoring item {item.id}: {e}")
            # Fallback to default score
            scored_item = ScoredNewsItem(
                **item.model_dump(),
                impact_score=3,
                impact_reason="Error during scoring",
                impact_dimensions=[],
            )
            return scored_item


async def score_impact_async(
    items: List[ClassifiedNewsItem],
    target_date: date,
    cache: Optional[NewsCache] = None,
    max_concurrent: int = 10,
) -> List[ScoredNewsItem]:
    """
    Score impact for classified news items with concurrent processing.
    
    Args:
        items: List of classified news items
        target_date: Target date for caching
        cache: Optional cache instance
        max_concurrent: Maximum number of concurrent API calls
        
    Returns:
        List of scored news items
    """
    prompt_template = load_prompt("impact_prompt")
    
    # Count cache hits
    cache_hits = 0
    if cache:
        for item in items:
            if cache.get_scored(item.id, target_date):
                cache_hits += 1
    
    print(f"Scoring impact for {len(items)} news items (concurrent={max_concurrent}, cache_hits={cache_hits})...")
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Process all items concurrently
    tasks = [
        _score_single_item(item, prompt_template, cache, target_date, semaphore)
        for item in items
    ]
    results = await async_tqdm.gather(*tasks, desc="Scoring impact")
    
    return list(results)


def score_impact(
    items: List[ClassifiedNewsItem],
    target_date: date | None = None,
    cache: Optional[NewsCache] = None,
    max_concurrent: int = 10,
) -> List[ScoredNewsItem]:
    """
    Score impact for classified news items (synchronous wrapper).
    
    Args:
        items: List of classified news items
        target_date: Target date for caching (defaults to today)
        cache: Optional cache instance
        max_concurrent: Maximum number of concurrent API calls
        
    Returns:
        List of scored news items
    """
    if target_date is None:
        from datetime import date as date_class
        target_date = date_class.today()
    
    return asyncio.run(score_impact_async(items, target_date, cache, max_concurrent))

