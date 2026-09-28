from __future__ import annotations

import multiprocessing as mp
import multiprocessing.pool
import time
from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

SEED = 42
DATA_FILE = "proteins.csv"
FEATURES = ["enzyme", "hydrofob"]
K_VALUES = range(1, 11)
MAX_ITERATIONS = 100
TOLERANCE = 1e-4
SCATTER_SAMPLE = 20_000
PROCESSES = mp.cpu_count()
CHUNKS_PER_PROCESS = 4

Points = npt.NDArray[np.float32]
Labels = npt.NDArray[np.int32]
Lengths = npt.NDArray[np.integer]
Chunks = list[tuple[int, int]]
Partial = tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], float]

worker_points: Points = np.empty((0, len(FEATURES)), np.float32)


def read_dataset() -> tuple[Points, Lengths]:
    frame = pd.read_csv(
        DATA_FILE,
        usecols=[*FEATURES, "sequence"],
        dtype={"enzyme": np.float32, "hydrofob": np.float32},
    )
    points: Points = frame[FEATURES].to_numpy(np.float32)
    lengths: Lengths = frame["sequence"].str.len().to_numpy(np.int64)
    return points, lengths


def assign(points: Points, centroids: Points) -> tuple[Labels, float]:
    labels = np.zeros(len(points), np.int32)
    best = np.full(len(points), np.inf, np.float32)
    for cluster, centroid in enumerate(centroids):
        distances = (points[:, 0] - centroid[0]) ** 2 + (
            points[:, 1] - centroid[1]
        ) ** 2
        closer = distances < best
        best[closer] = distances[closer]
        labels[closer] = cluster
    return labels, float(best.sum(dtype=np.float64))


def split(size: int, parts: int) -> Chunks:
    bounds = [round(part * size / parts) for part in range(parts + 1)]
    return list(pairwise(bounds))


def init_worker(points: Points) -> None:
    global worker_points  # noqa: PLW0603
    worker_points = points


def chunk_step(start: int, stop: int, centroids: Points) -> Partial:
    points = worker_points[start:stop]
    labels, inertia = assign(points, centroids)
    k = len(centroids)
    counts = np.bincount(labels, minlength=k)
    totals = np.empty((k, points.shape[1]))
    for feature in range(points.shape[1]):
        totals[:, feature] = np.bincount(
            labels, weights=points[:, feature], minlength=k
        )
    return counts, totals, inertia


def combine(results: list[Partial], centroids: Points) -> tuple[Points, float]:
    counts = np.sum([count for count, _, _ in results], axis=0)
    totals = np.sum([total for _, total, _ in results], axis=0)
    inertia = sum(value for _, _, value in results)
    nonempty = counts[:, None] > 0
    moved = np.where(nonempty, totals / np.maximum(counts, 1)[:, None], centroids)
    return moved.astype(np.float32), inertia


def kmeans(
    pool: multiprocessing.pool.Pool,
    chunks: Chunks,
    points: Points,
    k: int,
    rng: np.random.Generator,
) -> tuple[Points, float]:
    centroids = points[rng.choice(len(points), k, replace=False)]
    inertia = 0.0
    for _ in range(MAX_ITERATIONS):
        results = pool.starmap(
            chunk_step, [(start, stop, centroids) for start, stop in chunks]
        )
        moved, inertia = combine(results, centroids)
        if np.abs(moved - centroids).max() <= TOLERANCE:
            break
        centroids = moved
    return centroids, inertia


def elbow(
    pool: multiprocessing.pool.Pool,
    chunks: Chunks,
    points: Points,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    return np.array([kmeans(pool, chunks, points, k, rng)[1] for k in K_VALUES])


def optimal_k(inertias: npt.NDArray[np.float64]) -> int:
    ks = np.array(K_VALUES, np.float64)
    x = (ks - ks[0]) / (ks[-1] - ks[0])
    y = (inertias - inertias[-1]) / (inertias[0] - inertias[-1])
    # Distance from (x, y) to the line through (0, 1) and (1, 0), i.e. x + y = 1.
    distances = np.abs(x + y - 1) / np.sqrt(2)
    return int(ks[distances.argmax()])


def longest_sequence_cluster(
    labels: Labels, lengths: Lengths, centroids: Points
) -> tuple[int, int, float]:
    longest = int(lengths.max())
    candidates = np.unique(labels[lengths == longest])
    hydrofob = FEATURES.index("hydrofob")
    cluster = int(candidates[centroids[candidates, hydrofob].argmax()])
    average = float(lengths[labels == cluster].mean())
    return cluster, longest, average


def main() -> None:
    start = time.perf_counter()

    rng = np.random.default_rng(SEED)
    points, lengths = read_dataset()
    chunks = split(len(points), PROCESSES * CHUNKS_PER_PROCESS)
    # The pool is created once and holds the dataset for the whole run: making
    # one per k-means call would copy the data to the workers again and again.
    with mp.Pool(PROCESSES, initializer=init_worker, initargs=(points,)) as pool:
        inertias = elbow(pool, chunks, points, rng)
        k = optimal_k(inertias)
        centroids, _ = kmeans(pool, chunks, points, k, rng)
    labels, _ = assign(points, centroids)
    cluster, longest, average = longest_sequence_cluster(labels, lengths, centroids)

    elapsed = time.perf_counter() - start

    print(f"Number of processors: {PROCESSES}")
    print(f"Proteins read: {len(points)}")
    print(f"Optimum number of clusters (k): {k}")
    print("Cluster with the highest sequence length:")
    print(f"  id: {cluster}")
    print(f"  highest sequence length: {longest}")
    print(f"  average sequence length: {average:.2f}")
    print(
        f"  centroid: enzyme={centroids[cluster, 0]:.2f}, "
        f"hydrofob={centroids[cluster, 1]:.2f}"
    )
    print(f"Execution time: {elapsed:.3f} s")

    plt.figure("Elbow graph")
    plt.plot(K_VALUES, inertias, marker="o")
    plt.plot(k, inertias[k - K_VALUES.start], marker="o", color="red", markersize=12)
    plt.title(f"Elbow graph (optimum k = {k})")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.xticks(list(K_VALUES))
    plt.grid(visible=True, alpha=0.3)

    sample = rng.choice(len(points), min(SCATTER_SAMPLE, len(points)), replace=False)
    plt.figure("Clusters")
    plt.scatter(
        points[sample, 0], points[sample, 1], c=labels[sample], cmap="tab10", s=5
    )
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="black",
        marker="X",
        s=150,
        label="centroids",
    )
    plt.title(f"Clusters by enzyme and hydrofob (k = {len(centroids)})")
    plt.xlabel("enzyme")
    plt.ylabel("hydrofob")
    plt.legend()

    lowest = centroids.min(axis=0)
    span = np.maximum(centroids.max(axis=0) - lowest, 1e-12)
    plt.figure("Centroids heat map")
    plt.imshow((centroids - lowest) / span, cmap="viridis", aspect="auto")
    plt.colorbar(label="scaled value")
    plt.title("Heat map of the clusters' centroids")
    plt.xticks(range(len(FEATURES)), FEATURES)
    plt.yticks(range(len(centroids)), [f"cluster {i}" for i in range(len(centroids))])
    for cluster, centroid in enumerate(centroids):
        for feature, value in enumerate(centroid):
            plt.text(feature, cluster, f"{value:.2f}", ha="center", va="center")

    plt.show()


if __name__ == "__main__":
    main()
