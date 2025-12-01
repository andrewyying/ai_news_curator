"""Main pipeline entry point with timing and caching."""

import json
import time
from datetime import date
from pathlib import Path

from .fetchers import fetch_all_feeds
from .pipeline import (
    classify_zero_shot,
    score_impact,
    cluster_items,
    summarize_clusters,
    generate_markdown_report,
)
from .cache import NewsCache


def run_daily_pipeline(
    target_date: date | None = None,
    use_cache: bool = True,
    max_concurrent: int = 10,
) -> str:
    """
    Run the full daily news curation pipeline with timing and caching.

    Args:
        target_date: Target date for news (defaults to today)
        use_cache: Whether to use cache for processed items
        max_concurrent: Maximum number of concurrent API calls

    Returns:
        Path to generated report file
    """
    if target_date is None:
        target_date = date.today()

    # Initialize cache
    cache = NewsCache() if use_cache else None
    if cache:
        cache.clear_old_cache(days_to_keep=7)

    pipeline_start = time.time()
    print(f"Running daily pipeline for {target_date.isoformat()}")
    print("=" * 60)

    timing_stats = {}

    # Step 1: Fetch news
    print("\n[1/6] Fetching news from RSS feeds...")
    step_start = time.time()
    raw_items = fetch_all_feeds(target_date)
    timing_stats["fetch"] = time.time() - step_start
    print(f"  ✓ Fetched {len(raw_items)} items in {timing_stats['fetch']:.2f}s")

    if not raw_items:
        print("No news items fetched. Exiting.")
        return ""

    # Step 2: Classify
    print("\n[2/6] Classifying news items...")
    step_start = time.time()
    classified_items = classify_zero_shot(raw_items, target_date, cache, max_concurrent)
    timing_stats["classify"] = time.time() - step_start
    print(
        f"  ✓ Classified {len(classified_items)} items in {timing_stats['classify']:.2f}s"
    )

    # Step 3: Score impact
    print("\n[3/6] Scoring impact...")
    step_start = time.time()
    scored_items = score_impact(classified_items, target_date, cache, max_concurrent)
    timing_stats["score"] = time.time() - step_start
    print(f"  ✓ Scored {len(scored_items)} items in {timing_stats['score']:.2f}s")

    # Step 4: Cluster and deduplicate
    print("\n[4/6] Clustering and deduplicating...")
    step_start = time.time()
    clusters = cluster_items(scored_items)
    timing_stats["cluster"] = time.time() - step_start
    print(f"  ✓ Created {len(clusters)} clusters in {timing_stats['cluster']:.2f}s")

    # Step 5: Summarize
    print("\n[5/6] Generating summaries...")
    step_start = time.time()
    summarized_clusters = summarize_clusters(clusters, max_concurrent)
    timing_stats["summarize"] = time.time() - step_start
    print(
        f"  ✓ Summarized {len(summarized_clusters)} clusters in {timing_stats['summarize']:.2f}s"
    )

    # Step 6: Generate report
    print("\n[6/6] Generating markdown report...")
    step_start = time.time()
    markdown = generate_markdown_report(summarized_clusters, target_date)
    timing_stats["report"] = time.time() - step_start
    print(f"  ✓ Generated report in {timing_stats['report']:.2f}s")

    # Save curated data
    curated_dir = Path(__file__).parent.parent.parent / "data" / "curated"
    curated_dir.mkdir(parents=True, exist_ok=True)
    curated_file = curated_dir / f"{target_date.isoformat()}.curated.json"

    with open(curated_file, "w", encoding="utf-8") as f:
        json.dump(
            [c.model_dump(mode="json") for c in summarized_clusters],
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print(f"Saved curated data to {curated_file}")

    # Save report
    reports_dir = Path(__file__).parent.parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / f"{target_date.isoformat()}.md"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"Saved report to {report_file}")

    # Print timing summary
    total_time = time.time() - pipeline_start

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("\nTiming Summary:")
    print("-" * 60)
    for step, duration in timing_stats.items():
        percentage = (duration / total_time * 100) if total_time > 0 else 0
        print(f"  {step.capitalize():12s}: {duration:6.2f}s ({percentage:5.1f}%)")
    print(f"  {'Total':12s}: {total_time:6.2f}s")
    print("=" * 60)

    return str(report_file)
