import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import threading 
from functools import partial

def euclidean_distances(X, centroids):
    """
    Calculates Euclidean distance for the given arrays.
    Equation: d(x2, x1) = sqrt((x2 - x1)^2)
    """
    # Using broadcasting for efficient distance computation across all points
    return np.array([np.sqrt(np.sum((X - c)**2, axis=1)) for c in centroids]).T

def k_means(X, k, max_iters=100, tol=1e-4):
    """Custom K-means implementation from scratch."""
    # Initialize with random centroids from the dataset
    np.random.seed(42)  # Set seed for reproducibility across runs
    random_indices = np.random.choice(X.shape[0], size=k, replace=False)
    centroids = X[random_indices]
    
    for _ in range(max_iters):
        # Calculate distances using the euclidean formula
        distances = euclidean_distances(X, centroids)
        
        # Assign to nearest centroid
        labels = np.argmin(distances, axis=1)
        
        # Calculate new centroids
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i] 
            for i in range(k)
        ])
        
        # Check for convergence
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
            
        centroids = new_centroids
        
    # Calculate WCSS (Within-Cluster Sum of Squares) for the elbow graph
    wcss = np.sum([np.sum((X[labels == i] - centroids[i])**2) for i in range(k)])
    return centroids, labels, wcss

def find_optimal_k(wcss_list):
    """
    Automatically finds the "elbow" point by calculating the maximum 
    perpendicular distance from the curve to a line drawn from the 
    first to the last point of the WCSS values.
    """
    n_points = len(wcss_list)
    p1 = np.array([1, wcss_list[0]])
    p2 = np.array([n_points, wcss_list[-1]])
    
    max_dist = 0
    optimal_k = 1
    
    for i in range(n_points):
        p0 = np.array([i + 1, wcss_list[i]])
        # Perpendicular distance from p0 to the line connecting p1 and p2
        dist = np.abs((p2[0] - p1[0]) * (p1[1] - p0[1]) - (p2[1] - p1[1]) * (p1[0] - p0[0])) / np.linalg.norm(p2 - p1)
        if dist > max_dist:
            max_dist = dist
            optimal_k = i + 1
            
    return optimal_k

def main():
    # 1. Register start time
    start_time = time.time()
    
    # 2. Read the csv file using relative path
    df = pd.read_csv("proteins.csv")
    
    # Convert sequence to string (if not already) and get length to normalize data
    df['seq_length'] = df['sequence'].astype(str).apply(len)
    
    # Isolate clustering features
    X = df[['enzyme', 'hydrofob']].values
    
    # 3. Construct the elbow graph and find the optimal clusters number (k)
    K_MAX = 20
    wcss_list = []
    models = {}
    thread_results = {}

    def worker(k_val):
        # Wrapper function to capture the return values
        thread_results[k_val] = k_means(X, k_val)

    threads = []
    
    for k in range(1, K_MAX + 1):
        t = threading.Thread(target=worker, args=(k,))
        threads.append(t)

    for t in threads:
        t.start()
        
    for t in threads:
        t.join()

    # Reconstruct the models and wcss list in the correct order (1 to 10)
    for k in range(1, K_MAX + 1):
        centroids, labels, wcss = thread_results[k]
        wcss_list.append(wcss)
        models[k] = (centroids, labels)
        
    optimal_k = find_optimal_k(wcss_list)
    
    # 4. Cluster the data using the optimum value using k-means
    # (Reusing the already computed model for the optimal k to save time)
    best_centroids, best_labels = models[optimal_k]
    df['cluster'] = best_labels
    
    # 5. Find the cluster with the highest sequence length and compute its average
    # Identify the protein with the absolute highest sequence length
    max_seq_idx = df['seq_length'].idxmax()
    target_cluster = df.loc[max_seq_idx, 'cluster']
    highest_seq_val = df.loc[max_seq_idx, 'seq_length']
    
    # Compute the average sequence length for that specific cluster
    target_cluster_avg = df[df['cluster'] == target_cluster]['seq_length'].mean()
    
    # 6. Measure end time and print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.4f} seconds\n")
    
    # Print cluster results
    print(f"--- Cluster {target_cluster} Information ---")
    print(f"Highest Sequence Length Found: {highest_seq_val}")
    print(f"Average Sequence Length of Cluster {target_cluster}: {target_cluster_avg:.2f}")

    # 7. Plot results of the execution
    # Creating figures without stopping execution; will be shown simultaneously at the end
    print("\nGenerating plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot A: Elbow Graph
    axes[0].plot(range(1, K_MAX + 1), wcss_list, marker='o', linestyle='-', color='b', label='WCSS')
    
    # Define points for the reference line and optimal k
    p1 = np.array([1, wcss_list[0]])
    p2 = np.array([K_MAX, wcss_list[-1]])
    p0 = np.array([optimal_k, wcss_list[optimal_k - 1]])
    
    # Calculate the perpendicular projection point on the line
    line_vec = p2 - p1
    p0_vec = p0 - p1
    t = np.dot(p0_vec, line_vec) / np.dot(line_vec, line_vec)
    proj_point = p1 + t * line_vec
    
    # Plot the secant line between k=1 and k_Max
    axes[0].plot([p1[0], p2[0]], [p1[1], p2[1]], color='green', linestyle='--', label='Reference Line')
    
    # Plot the perpendicular distance line from the elbow to the secant line
    axes[0].plot([p0[0], proj_point[0]], [p0[1], proj_point[1]], color='red', linestyle='-', linewidth=2, label='Max Distance')
    
    # Highlight the elbow point
    axes[0].scatter(*p0, color='red', s=100, zorder=5, label=f'Optimal k={optimal_k}')
    
    axes[0].set_title('Elbow Graph')
    axes[0].set_xlabel('Number of clusters (k)')
    axes[0].set_ylabel('Within-Cluster Sum of Squares (WCSS)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)   
    
    # Plot B: Clusters with centroids
    sns.scatterplot(x='enzyme', y='hydrofob', hue='cluster', data=df, 
                    palette='tab10', ax=axes[1], s=15, alpha=0.5, legend='full')
    axes[1].scatter(best_centroids[:, 0], best_centroids[:, 1], 
                    c='black', s=200, marker='X', label='Centroids')
    axes[1].set_title(f'Data Clustered (k={optimal_k})')
    axes[1].legend()
    
    # Plot C: Heat map using the values of the clusters' centroids
    centroid_df = pd.DataFrame(best_centroids, columns=['enzyme', 'hydrofob'])
    centroid_df.index.name = 'Cluster ID'
    sns.heatmap(centroid_df, annot=True, cmap='coolwarm', ax=axes[2], fmt=".3f", cbar=True)
    axes[2].set_title('Centroid Values Heatmap')
    
    plt.tight_layout()
    # plt.show() blocks execution only at the very end, as requested
    plt.show()

if __name__ == "__main__":
    main()