"""Deduplication and clustering module."""

import uuid
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from models import ScoredNewsItem, ClusteredItem
from llm import embed_texts
from config import settings


def cluster_items(
    items: List[ScoredNewsItem],
    similarity_threshold: float | None = None,
) -> List[ClusteredItem]:
    """
    Cluster similar news items using embeddings.
    
    Args:
        items: List of scored news items
        similarity_threshold: Cosine similarity threshold (defaults to config value)
        
    Returns:
        List of clustered items
    """
    if similarity_threshold is None:
        similarity_threshold = settings.similarity_threshold
    
    if not items:
        return []
    
    print(f"Clustering {len(items)} news items...")
    
    # Prepare texts for embedding
    texts = []
    for item in items:
        text = f"{item.title}\n{item.content[:1000]}"
        texts.append(text)
    
    # Generate embeddings in batches
    batch_size = 100
    embeddings = []
    print("Generating embeddings...")
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        batch_embeddings = embed_texts(batch)
        embeddings.extend(batch_embeddings)
    
    embeddings = np.array(embeddings)
    
    # Simple clustering: iterate and assign to clusters
    clusters: List[ClusteredItem] = []
    cluster_representatives: List[ScoredNewsItem] = []
    cluster_embeddings: List[np.ndarray] = []
    
    for idx, item in enumerate(tqdm(items, desc="Clustering")):
        embedding = embeddings[idx]
        
        if not clusters:
            # First item creates first cluster
            cluster_id = str(uuid.uuid4())
            clusters.append(
                ClusteredItem(
                    cluster_id=cluster_id,
                    representative=item,
                    members=[item],
                )
            )
            cluster_representatives.append(item)
            cluster_embeddings.append(embedding)
        else:
            # Find best matching cluster
            similarities = cosine_similarity(
                embedding.reshape(1, -1),
                np.array(cluster_embeddings),
            )[0]
            
            max_similarity_idx = np.argmax(similarities)
            max_similarity = similarities[max_similarity_idx]
            
            if max_similarity >= similarity_threshold:
                # Add to existing cluster
                cluster = clusters[max_similarity_idx]
                cluster.members.append(item)
                
                # Update representative if this item has higher impact score
                if item.impact_score > cluster.representative.impact_score:
                    cluster.representative = item
                    cluster_embeddings[max_similarity_idx] = embedding
            else:
                # Create new cluster
                cluster_id = str(uuid.uuid4())
                clusters.append(
                    ClusteredItem(
                        cluster_id=cluster_id,
                        representative=item,
                        members=[item],
                    )
                )
                cluster_representatives.append(item)
                cluster_embeddings.append(embedding)
    
    print(f"Created {len(clusters)} clusters from {len(items)} items")
    
    return clusters

