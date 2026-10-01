from __future__ import annotations

import multiprocessing as mp
import time

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
PROCESSES = min(mp.cpu_count(), len(K_VALUES))

Points = npt.NDArray[np.float32]
Labels = npt.NDArray[np.int32]
Lengths = npt.NDArray[np.integer]
Model = tuple[Points, float]

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


def standardize(points: Points) -> tuple[Points, Points, Points]:
    # The features live on very different scales (enzyme 1-10, hydrofob
    # 40-220), so without this hydrofob would carry nearly all of the
    # Euclidean distance and enzyme would hardly affect the clusters.
    center = points.mean(axis=0, dtype=np.float64).astype(np.float32)
    spread = np.maximum(points.std(axis=0, dtype=np.float64), 1e-12).astype(np.float32)
    return (points - center) / spread, center, spread


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


def update(points: Points, labels: Labels, centroids: Points) -> Points:
    k = len(centroids)
    counts = np.bincount(labels, minlength=k)
    moved = np.empty_like(centroids)
    for feature in range(points.shape[1]):
        totals = np.bincount(labels, weights=points[:, feature], minlength=k)
        moved[:, feature] = np.where(
            counts > 0, totals / np.maximum(counts, 1), centroids[:, feature]
        )
    return moved


def kmeans(points: Points, k: int, rng: np.random.Generator) -> Model:
    centroids = points[rng.choice(len(points), k, replace=False)]
    labels, inertia = assign(points, centroids)
    for _ in range(MAX_ITERATIONS):
        moved = update(points, labels, centroids)
        if np.abs(moved - centroids).max() <= TOLERANCE:
            break
        centroids = moved
        labels, inertia = assign(points, centroids)
    return centroids, inertia


def init_worker(points: Points) -> None:
    global worker_points  # noqa: PLW0603
    worker_points = points


def kmeans_task(k: int) -> Model:
    return kmeans(worker_points, k, np.random.default_rng(SEED))


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

    points, lengths = read_dataset()
    scaled, center, spread = standardize(points)

    with mp.Pool(PROCESSES, initializer=init_worker, initargs=(scaled,)) as pool:
        models = pool.map(kmeans_task, K_VALUES)

    inertias = np.array([inertia for _, inertia in models])
    k = optimal_k(inertias)
    centroids = models[k - K_VALUES.start][0]
    labels, _ = assign(scaled, centroids)
    centroids = centroids * spread + center
    cluster, longest, average = longest_sequence_cluster(labels, lengths, centroids)

    elapsed = time.perf_counter() - start

    print(f"Number of processors: {mp.cpu_count()}")
    print(f"Worker processes: {PROCESSES}")
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

    optimum = float(inertias[k - K_VALUES.start])
    first, last = K_VALUES.start, K_VALUES[-1]
    lowest, highest = float(inertias[-1]), float(inertias[0])
    x = (k - first) / (last - first)
    y = (float(inertias[k - first]) - lowest) / (highest - lowest)
    # Foot of the perpendicular from (x, y) to the line x + y = 1.
    offset = (x + y - 1) / 2
    foot_k, foot_inertia = first + (x - offset) * (last - first), lowest + (y - offset) * (highest - lowest)
    plt.figure("Elbow graph")
    plt.plot(K_VALUES, inertias, marker="o", label="inertia")
    plt.plot(
        [K_VALUES.start, K_VALUES[-1]],
        [inertias[0], inertias[-1]],
        linestyle="--",
        color="gray",
        label="line joining the ends of the curve",
    )
    plt.plot(
        [k, foot_k],
        [optimum, foot_inertia],
        linestyle=":",
        color="red",
        label="distance from the curve to that line",
    )
    plt.plot(
        k,
        optimum,
        marker="o",
        color="red",
        markersize=12,
        label=f"optimum k = {k}",
    )
    plt.title(f"Elbow graph (optimum k = {k})")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.xticks(list(K_VALUES))
    plt.grid(visible=True, alpha=0.3)
    plt.legend()
    # The perpendicular is computed with both axes scaled to [0, 1], so it only
    # shows as a right angle if the box it is drawn in is square.
    plt.gca().set_box_aspect(1)

    rng = np.random.default_rng(SEED)
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
