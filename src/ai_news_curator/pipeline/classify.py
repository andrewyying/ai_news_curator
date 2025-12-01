"""News classification module with concurrent processing and caching."""

import asyncio
import json
from datetime import date
from typing import List, Optional
from tqdm.asyncio import tqdm as async_tqdm

from ..models import RawNewsItem, ClassifiedNewsItem
from ..llm import call_llm_json, load_prompt
from ..cache import NewsCache


CATEGORIES = [
    "AI Models",
    "AI Infrastructure & Hardware",
    "AI Research",
    "AI Policy & Regulation",
    "Developer Tools & Platforms",
    "Tech Business & Strategy",
    "Other",
]


async def _classify_single_item(
    item: RawNewsItem,
    prompt_template: str,
    cache: Optional[NewsCache],
    target_date: date,
    semaphore: asyncio.Semaphore,
) -> ClassifiedNewsItem:
    """Classify a single news item asynchronously."""
    async with semaphore:
        # Check cache first
        if cache:
            cached = cache.get_classified(item.id, target_date)
            if cached:
                return cached
        
        try:
            # Prepare input JSON (reduced content length for efficiency)
            input_data = {
                "title": item.title,
                "content": item.content[:500],  # Reduced from 2000 to 500 for faster processing
            }
            input_json = json.dumps(input_data, ensure_ascii=False)
            
            # Construct prompt
            prompt = f"{prompt_template}\n\nInput:\n{input_json}"
            
            # Call LLM (run in thread pool to avoid blocking)
            response = await asyncio.to_thread(call_llm_json, prompt, temperature=0.3)
            
            category = response.get("category", "Other")
            confidence = float(response.get("confidence", 0.5))
            
            # Validate category
            if category not in CATEGORIES:
                category = "Other"
            
            # Create classified item
            classified_item = ClassifiedNewsItem(
                **item.model_dump(),
                category=category,
                classification_confidence=confidence,
                classification_method="zero-shot",
            )
            
            # Save to cache
            if cache:
                cache.save_classified(classified_item, target_date)
            
            return classified_item
        
        except Exception as e:
            print(f"Error classifying item {item.id}: {e}")
            # Fallback to "Other" category
            classified_item = ClassifiedNewsItem(
                **item.model_dump(),
                category="Other",
                classification_confidence=0.0,
                classification_method="zero-shot",
            )
            return classified_item


async def classify_zero_shot_async(
    items: List[RawNewsItem],
    target_date: date,
    cache: Optional[NewsCache] = None,
    max_concurrent: int = 10,
) -> List[ClassifiedNewsItem]:
    """
    Classify news items using zero-shot LLM classification with concurrent processing.
    
    Args:
        items: List of raw news items
        target_date: Target date for caching
        cache: Optional cache instance
        max_concurrent: Maximum number of concurrent API calls
        
    Returns:
        List of classified news items
    """
    prompt_template = load_prompt("classifier_prompt_zero_shot")
    
    # Count cache hits
    cache_hits = 0
    if cache:
        for item in items:
            if cache.get_classified(item.id, target_date):
                cache_hits += 1
    
    print(f"Classifying {len(items)} news items (zero-shot, concurrent={max_concurrent}, cache_hits={cache_hits})...")
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Process all items concurrently
    tasks = [
        _classify_single_item(item, prompt_template, cache, target_date, semaphore)
        for item in items
    ]
    results = await async_tqdm.gather(*tasks, desc="Classifying")
    
    return list(results)


def classify_zero_shot(
    items: List[RawNewsItem],
    target_date: date | None = None,
    cache: Optional[NewsCache] = None,
    max_concurrent: int = 10,
) -> List[ClassifiedNewsItem]:
    """
    Classify news items using zero-shot LLM classification (synchronous wrapper).
    
    Args:
        items: List of raw news items
        target_date: Target date for caching (defaults to today)
        cache: Optional cache instance
        max_concurrent: Maximum number of concurrent API calls
        
    Returns:
        List of classified news items
    """
    if target_date is None:
        from datetime import date as date_class
        target_date = date_class.today()
    
    return asyncio.run(classify_zero_shot_async(items, target_date, cache, max_concurrent))


async def _classify_single_item_few_shot(
    item: RawNewsItem,
    prompt_template: str,
    semaphore: asyncio.Semaphore,
) -> ClassifiedNewsItem:
    """Classify a single news item using few-shot asynchronously."""
    async with semaphore:
        try:
            # Prepare input JSON (reduced content length for efficiency)
            input_data = {
                "title": item.title,
                "content": item.content[:500],  # Reduced from 2000 to 500 for faster processing
            }
            input_json = json.dumps(input_data, ensure_ascii=False)
            
            # Construct prompt
            prompt = f"{prompt_template}\n\nInput:\n{input_json}"
            
            # Call LLM (run in thread pool to avoid blocking)
            response = await asyncio.to_thread(call_llm_json, prompt, temperature=0.3)
            
            category = response.get("category", "Other")
            confidence = float(response.get("confidence", 0.5))
            
            # Validate category
            if category not in CATEGORIES:
                category = "Other"
            
            # Create classified item
            return ClassifiedNewsItem(
                **item.model_dump(),
                category=category,
                classification_confidence=confidence,
                classification_method="few-shot",
            )
        
        except Exception as e:
            print(f"Error classifying item {item.id}: {e}")
            # Fallback to "Other" category
            return ClassifiedNewsItem(
                **item.model_dump(),
                category="Other",
                classification_confidence=0.0,
                classification_method="few-shot",
            )


async def classify_few_shot_async(
    items: List[RawNewsItem],
    max_concurrent: int = 10,
) -> List[ClassifiedNewsItem]:
    """
    Classify news items using few-shot LLM classification with concurrent processing.
    
    Args:
        items: List of raw news items
        max_concurrent: Maximum number of concurrent API calls
        
    Returns:
        List of classified news items
    """
    prompt_template = load_prompt("classifier_prompt_few_shot")
    
    print(f"Classifying {len(items)} news items (few-shot, concurrent={max_concurrent})...")
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Process all items concurrently
    tasks = [
        _classify_single_item_few_shot(item, prompt_template, semaphore)
        for item in items
    ]
    results = await async_tqdm.gather(*tasks, desc="Classifying")
    
    return list(results)


def classify_few_shot(
    items: List[RawNewsItem],
    max_concurrent: int = 10,
) -> List[ClassifiedNewsItem]:
    """
    Classify news items using few-shot LLM classification (synchronous wrapper).
    
    Args:
        items: List of raw news items
        max_concurrent: Maximum number of concurrent API calls
        
    Returns:
        List of classified news items
    """
    return asyncio.run(classify_few_shot_async(items, max_concurrent))

