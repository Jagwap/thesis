import numpy as np

from sklearn.cluster import KMeans

def split_into_chunks(positions,n_clusters = 3):
    X = np.vstack(positions)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels

def find_clusters(positions,n_clusters = 3):
    X = np.vstack(positions)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    cluster_means = []
    for cluster_id in range(n_clusters):
        cluster_points = X[labels == cluster_id]
        cluster_mean = cluster_points.mean(axis=0)
        cluster_means.append(cluster_mean)
    cluster_means = np.vstack(cluster_means)
    return cluster_means


def fps(values,target):
    chosen = [values[0], values[-1]]
    remaining = set(values[1:-1])
    res = []
    while len(chosen) < target and remaining:
        best = max(remaining, key=lambda x: min(abs(x - c) for c in chosen))
        res.append(values.index(best))
        chosen.append(best)
        remaining.remove(best)

    return sorted(res)