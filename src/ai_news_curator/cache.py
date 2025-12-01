"""Cache management for processed news items."""

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from .models import ClassifiedNewsItem, ScoredNewsItem


class NewsCache:
    """Cache for processed news items to avoid reprocessing."""
    
    def __init__(self, cache_dir: Path | None = None):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory to store cache files (defaults to data/cache)
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "data" / "cache"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, item_id: str, operation: str, date_str: str) -> str:
        """Generate cache key."""
        content = f"{item_id}:{operation}:{date_str}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_file(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.json"
    
    def get_classified(self, item_id: str, target_date: date) -> Optional[ClassifiedNewsItem]:
        """
        Get cached classification result.
        
        Args:
            item_id: News item ID
            target_date: Target date
            
        Returns:
            Cached ClassifiedNewsItem or None
        """
        cache_key = self._get_cache_key(item_id, "classify", target_date.isoformat())
        cache_file = self._get_cache_file(cache_key)
        
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return ClassifiedNewsItem(**data)
            except Exception:
                return None
        return None
    
    def save_classified(self, item: ClassifiedNewsItem, target_date: date):
        """
        Save classification result to cache.
        
        Args:
            item: ClassifiedNewsItem to cache
            target_date: Target date
        """
        cache_key = self._get_cache_key(item.id, "classify", target_date.isoformat())
        cache_file = self._get_cache_file(cache_key)
        
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(item.model_dump(mode="json"), f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Failed to save cache for {item.id}: {e}")
    
    def get_scored(self, item_id: str, target_date: date) -> Optional[ScoredNewsItem]:
        """
        Get cached scoring result.
        
        Args:
            item_id: News item ID
            target_date: Target date
            
        Returns:
            Cached ScoredNewsItem or None
        """
        cache_key = self._get_cache_key(item_id, "score", target_date.isoformat())
        cache_file = self._get_cache_file(cache_key)
        
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return ScoredNewsItem(**data)
            except Exception:
                return None
        return None
    
    def save_scored(self, item: ScoredNewsItem, target_date: date):
        """
        Save scoring result to cache.
        
        Args:
            item: ScoredNewsItem to cache
            target_date: Target date
        """
        cache_key = self._get_cache_key(item.id, "score", target_date.isoformat())
        cache_file = self._get_cache_file(cache_key)
        
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(item.model_dump(mode="json"), f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Failed to save cache for {item.id}: {e}")
    
    def clear_old_cache(self, days_to_keep: int = 7):
        """
        Clear cache files older than specified days.
        
        Args:
            days_to_keep: Number of days to keep cache files
        """
        cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
            except Exception:
                pass

