"""Configuration management for AI News Curator."""

import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_api_base: str = Field(default="https://api.openai.com/v1", description="OpenAI API base URL")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model name")
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model name")
    
    # News filtering
    max_news_age_days: int = Field(default=2, description="Maximum age of news in days")
    
    # Clustering
    similarity_threshold: float = Field(default=0.8, description="Similarity threshold for clustering")
    
    rss_feeds: List[str] = Field(
        default=[
            "https://hnrss.org/frontpage",
            "https://techcrunch.com/feed/",
            "https://openai.com/blog/rss.xml",
            "https://developer.nvidia.com/blog/feed/",
            "http://export.arxiv.org/rss/cs.CL",
            "http://export.arxiv.org/rss/cs.AI",
        ],
        description="List of RSS feed URLs"
    )
    
    @field_validator("rss_feeds", mode="before")
    @classmethod
    def parse_rss_feeds(cls, v):
        """Parse RSS_FEEDS from environment variable or use defaults."""
        if isinstance(v, str):
            # Parse comma-separated string
            return [url.strip() for url in v.split(",") if url.strip()]
        return v


# Global settings instance
try:
    settings = Settings()
except Exception as e:
    print(f"Warning: Failed to load settings: {e}")
    print("Please create a .env file with OPENAI_API_KEY")
    # Create a minimal settings object for development
    settings = Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    )

