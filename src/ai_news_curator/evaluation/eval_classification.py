"""Classification evaluation script comparing zero-shot vs few-shot."""

import json
from datetime import date
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

from ..models import RawNewsItem
from ..pipeline.classify import classify_zero_shot, classify_few_shot, CATEGORIES


def load_sample_labels() -> List[Dict]:
    """Load sample labels from JSON file."""
    labels_file = Path(__file__).parent / "sample_labels.json"
    with open(labels_file, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_classification():
    """Evaluate zero-shot vs few-shot classification accuracy."""
    print("Loading sample labels...")
    labels = load_sample_labels()
    
    # Convert to RawNewsItem
    raw_items = []
    true_labels = {}
    
    for label in labels:
        item = RawNewsItem(
            id=label["id"],
            title=label["title"],
            content=label["content"],
            source="evaluation",
        )
        raw_items.append(item)
        true_labels[label["id"]] = label["true_category"]
    
    print(f"Loaded {len(raw_items)} labeled samples")
    print("\n" + "=" * 60)
    
    # Use a dummy date for evaluation (cache will be based on this date)
    eval_date = date.today()
    
    # Run zero-shot classification (with concurrency, no cache for evaluation)
    print("\n[1/2] Running zero-shot classification...")
    zero_shot_results = classify_zero_shot(raw_items, eval_date, None, max_concurrent=10)
    
    # Run few-shot classification (with concurrency)
    print("\n[2/2] Running few-shot classification...")
    few_shot_results = classify_few_shot(raw_items, max_concurrent=10)
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # Zero-shot metrics
    zero_shot_correct = sum(
        1 for item in zero_shot_results
        if item.category == true_labels[item.id]
    )
    zero_shot_accuracy = zero_shot_correct / len(zero_shot_results) if zero_shot_results else 0
    
    # Few-shot metrics
    few_shot_correct = sum(
        1 for item in few_shot_results
        if item.category == true_labels[item.id]
    )
    few_shot_accuracy = few_shot_correct / len(few_shot_results) if few_shot_results else 0
    
    # Per-category metrics
    zero_shot_by_category = defaultdict(lambda: {"correct": 0, "total": 0})
    few_shot_by_category = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for item in zero_shot_results:
        true_cat = true_labels[item.id]
        zero_shot_by_category[true_cat]["total"] += 1
        if item.category == true_cat:
            zero_shot_by_category[true_cat]["correct"] += 1
    
    for item in few_shot_results:
        true_cat = true_labels[item.id]
        few_shot_by_category[true_cat]["total"] += 1
        if item.category == true_cat:
            few_shot_by_category[true_cat]["correct"] += 1
    
    # Print results
    print(f"\nOverall Accuracy:")
    print(f"  Zero-shot: {zero_shot_accuracy:.2%} ({zero_shot_correct}/{len(zero_shot_results)})")
    print(f"  Few-shot:  {few_shot_accuracy:.2%} ({few_shot_correct}/{len(few_shot_results)})")
    print(f"  Improvement: {few_shot_accuracy - zero_shot_accuracy:+.2%}")
    
    print(f"\nPer-Category Accuracy:")
    print(f"\n{'Category':<35} {'Zero-shot':<15} {'Few-shot':<15}")
    print("-" * 65)
    
    all_categories = set(zero_shot_by_category.keys()) | set(few_shot_by_category.keys())
    for cat in sorted(all_categories):
        zs_total = zero_shot_by_category[cat]["total"]
        zs_correct = zero_shot_by_category[cat]["correct"]
        zs_acc = zs_correct / zs_total if zs_total > 0 else 0
        
        fs_total = few_shot_by_category[cat]["total"]
        fs_correct = few_shot_by_category[cat]["correct"]
        fs_acc = fs_correct / fs_total if fs_total > 0 else 0
        
        print(f"{cat:<35} {zs_acc:>6.2%} ({zs_correct}/{zs_total}) {fs_acc:>6.2%} ({fs_correct}/{fs_total})")
    
    # Save results to markdown
    results_file = Path(__file__).parent / "classification_results.md"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("# Classification Evaluation Results\n\n")
        f.write("## Overall Accuracy\n\n")
        f.write(f"- **Zero-shot:** {zero_shot_accuracy:.2%} ({zero_shot_correct}/{len(zero_shot_results)})\n")
        f.write(f"- **Few-shot:** {few_shot_accuracy:.2%} ({few_shot_correct}/{len(few_shot_results)})\n")
        f.write(f"- **Improvement:** {few_shot_accuracy - zero_shot_accuracy:+.2%}\n\n")
        
        f.write("## Per-Category Accuracy\n\n")
        f.write("| Category | Zero-shot | Few-shot |\n")
        f.write("|----------|-----------|----------|\n")
        
        for cat in sorted(all_categories):
            zs_total = zero_shot_by_category[cat]["total"]
            zs_correct = zero_shot_by_category[cat]["correct"]
            zs_acc = zs_correct / zs_total if zs_total > 0 else 0
            
            fs_total = few_shot_by_category[cat]["total"]
            fs_correct = few_shot_by_category[cat]["correct"]
            fs_acc = fs_correct / fs_total if fs_total > 0 else 0
            
            f.write(f"| {cat} | {zs_acc:.2%} ({zs_correct}/{zs_total}) | {fs_acc:.2%} ({fs_correct}/{fs_total}) |\n")
        
        f.write("\n## Detailed Predictions\n\n")
        f.write("### Zero-shot Predictions\n\n")
        f.write("| ID | Title | True Label | Predicted | Correct |\n")
        f.write("|----|-------|------------|-----------|--------|\n")
        
        for item in zero_shot_results:
            true_cat = true_labels[item.id]
            correct = "✓" if item.category == true_cat else "✗"
            f.write(f"| {item.id} | {item.title[:50]} | {true_cat} | {item.category} | {correct} |\n")
        
        f.write("\n### Few-shot Predictions\n\n")
        f.write("| ID | Title | True Label | Predicted | Correct |\n")
        f.write("|----|-------|------------|-----------|--------|\n")
        
        for item in few_shot_results:
            true_cat = true_labels[item.id]
            correct = "✓" if item.category == true_cat else "✗"
            f.write(f"| {item.id} | {item.title[:50]} | {true_cat} | {item.category} | {correct} |\n")
    
    print(f"\n✅ Results saved to {results_file}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    evaluate_classification()

