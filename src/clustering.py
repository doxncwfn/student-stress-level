"""
Clustering Module
Implements various clustering algorithms for stress pattern analysis
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')


class StressClusterer:
    """Class to handle clustering analysis for stress patterns"""
    
    def __init__(self):
        """Initialize clustering algorithms"""
        self.models = {}
        self.labels = {}
        self.metrics = {}
        
    def kmeans_clustering(self, X, n_clusters=3, random_state=42):
        """
        Perform K-Means clustering
        
        Args:
            X: Features
            n_clusters (int): Number of clusters
            random_state (int): Random seed
        
        Returns:
            array: Cluster labels
        """
        print(f"\nPerforming K-Means Clustering (k={n_clusters})...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        
        self.models['KMeans'] = kmeans
        self.labels['KMeans'] = labels
        
        # Calculate metrics
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        
        self.metrics['KMeans'] = {
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'calinski_harabasz_score': calinski_harabasz,
            'inertia': kmeans.inertia_,
            'n_clusters': n_clusters
        }
        
        print(f"✓ K-Means clustering completed")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Davies-Bouldin Score: {davies_bouldin:.4f}")
        print(f"  Calinski-Harabasz Score: {calinski_harabasz:.4f}")
        
        return labels
    
    def hierarchical_clustering(self, X, n_clusters=3, linkage_method='ward'):
        """
        Perform Hierarchical Clustering
        
        Args:
            X: Features
            n_clusters (int): Number of clusters
            linkage_method (str): Linkage method ('ward', 'complete', 'average', 'single')
        
        Returns:
            array: Cluster labels
        """
        print(f"\nPerforming Hierarchical Clustering (k={n_clusters}, linkage={linkage_method})...")
        
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        labels = hierarchical.fit_predict(X)
        
        self.models['Hierarchical'] = hierarchical
        self.labels['Hierarchical'] = labels
        
        # Calculate metrics
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        
        self.metrics['Hierarchical'] = {
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'calinski_harabasz_score': calinski_harabasz,
            'n_clusters': n_clusters,
            'linkage': linkage_method
        }
        
        print(f"✓ Hierarchical clustering completed")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Davies-Bouldin Score: {davies_bouldin:.4f}")
        print(f"  Calinski-Harabasz Score: {calinski_harabasz:.4f}")
        
        return labels
    
    def dbscan_clustering(self, X, eps=0.5, min_samples=5):
        """
        Perform DBSCAN clustering
        
        Args:
            X: Features
            eps (float): Maximum distance between samples
            min_samples (int): Minimum samples in a neighborhood
        
        Returns:
            array: Cluster labels
        """
        print(f"\nPerforming DBSCAN Clustering (eps={eps}, min_samples={min_samples})...")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        self.models['DBSCAN'] = dbscan
        self.labels['DBSCAN'] = labels
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Calculate metrics only if we have more than 1 cluster
        if n_clusters > 1:
            # Filter out noise points for metric calculation
            mask = labels != -1
            if mask.sum() > 0:
                silhouette = silhouette_score(X[mask], labels[mask])
                davies_bouldin = davies_bouldin_score(X[mask], labels[mask])
                calinski_harabasz = calinski_harabasz_score(X[mask], labels[mask])
            else:
                silhouette = davies_bouldin = calinski_harabasz = 0
        else:
            silhouette = davies_bouldin = calinski_harabasz = 0
        
        self.metrics['DBSCAN'] = {
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'calinski_harabasz_score': calinski_harabasz,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'eps': eps,
            'min_samples': min_samples
        }
        
        print(f"✓ DBSCAN clustering completed")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Number of noise points: {n_noise}")
        if n_clusters > 1:
            print(f"  Silhouette Score: {silhouette:.4f}")
            print(f"  Davies-Bouldin Score: {davies_bouldin:.4f}")
            print(f"  Calinski-Harabasz Score: {calinski_harabasz:.4f}")
        
        return labels
    
    def find_optimal_k(self, X, k_range=range(2, 11)):
        """
        Find optimal number of clusters using elbow method and silhouette score
        
        Args:
            X: Features
            k_range: Range of k values to test
        
        Returns:
            dict: Metrics for each k value
        """
        print("\nFinding optimal number of clusters...")
        
        results = {
            'k': [],
            'inertia': [],
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': []
        }
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            results['k'].append(k)
            results['inertia'].append(kmeans.inertia_)
            results['silhouette'].append(silhouette_score(X, labels))
            results['davies_bouldin'].append(davies_bouldin_score(X, labels))
            results['calinski_harabasz'].append(calinski_harabasz_score(X, labels))
        
        df = pd.DataFrame(results)
        
        print("\n" + "="*70)
        print("OPTIMAL K ANALYSIS")
        print("="*70)
        print(df.to_string(index=False))
        
        # Find optimal k based on silhouette score
        optimal_k = df.loc[df['silhouette'].idxmax(), 'k']
        print(f"\n✓ Optimal k based on Silhouette Score: {int(optimal_k)}")
        
        return df
    
    def analyze_clusters(self, X, labels, feature_names=None):
        """
        Analyze cluster characteristics
        
        Args:
            X: Features
            labels: Cluster labels
            feature_names: List of feature names
        
        Returns:
            DataFrame: Cluster statistics
        """
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_names)
        
        # Add cluster labels
        X_clustered = X.copy()
        X_clustered['Cluster'] = labels
        
        # Calculate cluster statistics
        cluster_stats = X_clustered.groupby('Cluster').agg(['mean', 'std', 'count'])
        
        print("\n" + "="*70)
        print("CLUSTER ANALYSIS")
        print("="*70)
        print(f"\nCluster Distribution:")
        print(X_clustered['Cluster'].value_counts().sort_index())
        
        print(f"\nCluster Statistics (Mean values):")
        print(cluster_stats.xs('mean', axis=1, level=1))
        
        return cluster_stats
    
    def compare_clustering_methods(self):
        """
        Compare all clustering methods
        
        Returns:
            DataFrame: Comparison of clustering methods
        """
        if not self.metrics:
            print("✗ No clustering performed yet!")
            return None
        
        comparison = []
        for method, metrics in self.metrics.items():
            comparison.append({
                'Method': method,
                'N_Clusters': metrics.get('n_clusters', 'N/A'),
                'Silhouette': f"{metrics['silhouette_score']:.4f}" if metrics['silhouette_score'] > 0 else 'N/A',
                'Davies-Bouldin': f"{metrics['davies_bouldin_score']:.4f}" if metrics['davies_bouldin_score'] > 0 else 'N/A',
                'Calinski-Harabasz': f"{metrics['calinski_harabasz_score']:.2f}" if metrics['calinski_harabasz_score'] > 0 else 'N/A'
            })
        
        df = pd.DataFrame(comparison)
        
        print("\n" + "="*70)
        print("CLUSTERING METHODS COMPARISON")
        print("="*70)
        print(df.to_string(index=False))
        
        return df
    
    def get_cluster_labels(self, method='KMeans'):
        """
        Get cluster labels for a specific method
        
        Args:
            method (str): Clustering method name
        
        Returns:
            array: Cluster labels
        """
        if method not in self.labels:
            print(f"✗ No labels found for method '{method}'")
            return None
        
        return self.labels[method]
