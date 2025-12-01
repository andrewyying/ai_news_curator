"""RSS feed fetcher for news aggregation."""

import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Set
from urllib.parse import urlparse

import feedparser
import requests
from tqdm import tqdm

from ..config import settings
from ..models import RawNewsItem


def _generate_id(source: str, title: str, url: str | None) -> str:
    """Generate unique ID for a news item."""
    content = f"{source}:{title}:{url or ''}"
    return hashlib.md5(content.encode()).hexdigest()


def _parse_date(date_str: str | None) -> datetime | None:
    """Parse date string from RSS feed."""
    if not date_str:
        return None
    
    try:
        # Try parsing with feedparser's parsed date
        parsed = feedparser._parse_date(date_str)
        if parsed:
            return datetime(*parsed[:6])
    except (ValueError, TypeError):
        pass
    
    return None


def fetch_all_feeds(target_date: date | None = None) -> List[RawNewsItem]:
    """
    Fetch news from all configured RSS feeds.
    
    Args:
        target_date: Target date for news (defaults to today).
                    Only news from the last N days (max_news_age_days) will be kept.
    
    Returns:
        List of RawNewsItem objects
    """
    if target_date is None:
        target_date = date.today()
    
    cutoff_date = datetime.combine(
        target_date - timedelta(days=settings.max_news_age_days),
        datetime.min.time()
    )
    
    # Not caching feeds for the sake of simplicity
    all_items: List[RawNewsItem] = []
    seen_urls: Set[str] = set()
    
    feeds = settings.rss_feeds
    print(f"Fetching news from {len(feeds)} RSS feeds...")
    
    for feed_url in tqdm(feeds, desc="Fetching feeds"):
        try:
            # Fetch feed
            feed_url_str = str(feed_url)
            response = requests.get(feed_url_str, timeout=30)
            response.raise_for_status()
            
            # Parse feed
            feed = feedparser.parse(response.content)
            
            if feed.bozo and feed.bozo_exception:
                print(f"Warning: Error parsing feed {feed_url_str}: {feed.bozo_exception}")
                continue
            
            source_name = feed.feed.get("title", urlparse(feed_url_str).netloc)
            
            # Process entries
            for entry in feed.entries:
                # Extract URL
                url = entry.get("link")
                if url and url in seen_urls:
                    continue
                
                # Extract title
                title = entry.get("title", "").strip()
                if not title:
                    continue
                
                # Extract content
                content = ""
                if "content" in entry:
                    content = entry.content[0].get("value", "")
                elif "summary" in entry:
                    content = entry.summary
                elif "description" in entry:
                    content = entry.description
                
                # Extract published date
                published_at = None
                if "published_parsed" in entry and entry.published_parsed:
                    try:
                        published_at = datetime(*entry.published_parsed[:6])
                    except (ValueError, TypeError):
                        pass
                elif "published" in entry:
                    published_at = _parse_date(entry.published)
                
                # Filter by date
                if published_at and published_at < cutoff_date:
                    continue
                
                # Create news item
                news_id = _generate_id(source_name, title, url)
                item = RawNewsItem(
                    id=news_id,
                    title=title,
                    url=url,
                    source=source_name,
                    published_at=published_at,
                    content=content[:5000] if content else "",  # Limit content length
                )
                
                all_items.append(item)
                if url:
                    seen_urls.add(url)
        
        except requests.RequestException as e:
            print(f"Error fetching feed {feed_url_str}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error processing feed {feed_url_str}: {e}")
            continue
    
    print(f"Fetched {len(all_items)} unique news items")
    
    # Save raw news to file
    _save_raw_news(all_items, target_date)
    
    return all_items


def _save_raw_news(items: List[RawNewsItem], target_date: date):
    """Save raw news items to JSON file."""
    data_dir = Path(__file__).parent.parent.parent.parent / "data" / "raw_news"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    filename = data_dir / f"{target_date.isoformat()}.raw.json"
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            [item.model_dump(mode="json") for item in items],
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    
    print(f"Saved raw news to {filename}")

