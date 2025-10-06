import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Any


def find_cluster_centroids(embeddings, max_k=10) -> Any:
    inertia = []
    cluster_centroids = []
    silhouette_scores = []
    K = range(1, max_k+1)

    embeddings = np.asarray(embeddings)
    num_embeddings = len(embeddings)

    sample_indices = None
    sampled_embeddings = embeddings
    if num_embeddings > 1:
        # ``silhouette_score`` performs an expensive pairwise distance computation.
        # Sampling keeps the complexity manageable for long videos while
        # producing a stable estimate for the optimal cluster count.
        sample_size = min(120, num_embeddings)
        if num_embeddings > sample_size:
            sample_indices = np.linspace(0, num_embeddings - 1, sample_size, dtype=int)
            sampled_embeddings = embeddings[sample_indices]

    for idx, k in enumerate(K):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
        kmeans.fit(embeddings)
        inertia.append(kmeans.inertia_)
        cluster_centroids.append({"k": k, "centroids": kmeans.cluster_centers_})

        if k > 1:
            try:
                if sample_indices is None:
                    score_embeddings = embeddings
                    score_labels = kmeans.labels_
                else:
                    score_embeddings = sampled_embeddings
                    score_labels = kmeans.labels_[sample_indices]

                if len(np.unique(score_labels)) < 2:
                    continue

                score = silhouette_score(score_embeddings, score_labels)
            except ValueError:
                continue
            silhouette_scores.append({"index": idx, "score": score})

    if silhouette_scores:
        optimal_index = max(silhouette_scores, key=lambda x: x["score"])['index']
    elif len(inertia) > 1:
        diffs = [inertia[i] - inertia[i+1] for i in range(len(inertia)-1)]
        optimal_index = diffs.index(max(diffs)) + 1
    else:
        optimal_index = 0

    optimal_centroids = cluster_centroids[optimal_index]['centroids']

    return optimal_centroids

def find_closest_centroid(centroids: list, normed_face_embedding) -> list:
    try:
        centroids = np.array(centroids)
        normed_face_embedding = np.array(normed_face_embedding)
        similarities = np.dot(centroids, normed_face_embedding)
        closest_centroid_index = np.argmax(similarities)

        return closest_centroid_index, centroids[closest_centroid_index]
    except ValueError:
        return None
