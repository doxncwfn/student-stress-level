"""
Visualization Module
Creates various plots and charts for data analysis and results presentation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class DataVisualizer:
    """Class to handle all visualization tasks"""
    
    def __init__(self):
        """Initialize visualizer"""
        self.figures = []
        
    def plot_target_distribution(self, y, title="Stress Level Distribution"):
        """
        Plot distribution of target variable
        
        Args:
            y: Target variable
            title (str): Plot title
        """
        plt.figure(figsize=(10, 6))
        
        if isinstance(y, pd.Series):
            value_counts = y.value_counts().sort_index()
        else:
            value_counts = pd.Series(y).value_counts().sort_index()
        
        ax = value_counts.plot(kind='bar', color='steelblue', edgecolor='black')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Stress Level', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=0)
        
        # Add value labels on bars
        for i, v in enumerate(value_counts):
            ax.text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/target_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Target distribution plot saved to 'results/target_distribution.png'")
        plt.show()
        
    def plot_correlation_matrix(self, data, title="Feature Correlation Matrix"):
        """
        Plot correlation matrix heatmap
        
        Args:
            data: DataFrame with features
            title (str): Plot title
        """
        plt.figure(figsize=(14, 12))
        
        # Calculate correlation matrix
        corr = data.corr()
        
        # Create heatmap
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('results/correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Correlation matrix saved to 'results/correlation_matrix.png'")
        plt.show()
        
    def plot_high_corr_scatterplots(self, data, corr_threshold=0.7, title_prefix="High Correlation Scatterplots"):
        """
        Plot scatterplots for all highly correlated feature pairs (|r| >= threshold).

        Args:
            data: DataFrame of features.
            corr_threshold (float): Threshold for absolute correlation.
            title_prefix (str): Prefix for plot title.
        """
        import numpy as np

        corr_matrix = data.corr()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        high_corr_pairs = [
            (row, col, corr_matrix.loc[row, col])
            for row in upper_tri.index
            for col in upper_tri.columns
            if pd.notnull(upper_tri.loc[row, col]) and abs(upper_tri.loc[row, col]) >= corr_threshold
        ]

        if high_corr_pairs:
            nplots = len(high_corr_pairs)
            rows = int(np.ceil(nplots / 2))
            cols = 2 if nplots > 1 else 1

            fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), dpi=130, squeeze=False)

            for idx, (x, y, corr_val) in enumerate(high_corr_pairs):
                r, c = divmod(idx, cols)
                ax = axes[r, c]
                sns.scatterplot(
                    data=data,
                    x=x,
                    y=y,
                    ax=ax,
                    alpha=0.5,
                )
                sns.regplot(
                    data=data,
                    x=x,
                    y=y,
                    ax=ax,
                    scatter=False,
                    line_kws={'color': 'red', 'linewidth': 2}
                )
                ax.set_title(rf"{title_prefix}\n{x} vs {y}\n$r$ = {corr_val:.2f}", fontsize=12)
                ax.set_xlabel(x)
                ax.set_ylabel(y)

            # Hide unused axes
            for j in range(idx + 1, rows * cols):
                r, c = divmod(j, cols)
                axes[r, c].axis('off')

            plt.tight_layout()
            plt.savefig('results/high_corr_scatterplots.png', dpi=300, bbox_inches='tight')
            print(f"✓ High-correlation scatterplots saved to 'results/high_corr_scatterplots.png'")
            plt.show()
        else:
            print(f"No highly correlated pairs found (|r| ≥ {corr_threshold})")
        
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, title="Confusion Matrix"):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            title (str): Plot title
        """
        plt.figure(figsize=(10, 8))
        
        cm = confusion_matrix(y_true, y_pred)
        
        if labels is None:
            labels = sorted(list(set(y_true)))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={"shrink": 0.8})
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Confusion matrix saved to 'results/confusion_matrix.png'")
        plt.show()
        
    def plot_outlier_violinplots(self, X, groupby=None, selected_cols=None, title="Feature Distribution by Outlier/Group"):
        """
        Plot violin plots for each feature, optionally grouped by a variable (e.g., stress level).
        If groupby is not specified, plots violins for features in X.
        If selected_cols is None, all columns will be used.

        Args:
            X (pd.DataFrame): Feature dataframe.
            groupby (str, optional): Column to group violins by (e.g., 'stress_level').
            selected_cols (list, optional): Specific feature columns to plot.
            title (str): Overall plot title.
        """
        import math

        df = X.copy()
        # Validate groupby column if specified
        if groupby is not None:
            if groupby not in df.columns:
                raise ValueError(f"Grouping column '{groupby}' not found in dataframe.")

        # Auto-select numeric columns, excluding groupby if necessary
        if selected_cols is None:
            selected_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if groupby in selected_cols:
                selected_cols.remove(groupby)

        n_features = len(selected_cols)
        n_cols = 3
        n_rows = int(np.ceil(n_features / n_cols)) if n_features > 0 else 0

        if n_features == 0:
            print("No features available for violin plot.")
            return

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows), dpi=150)
        axes = np.array(axes).reshape(n_rows, n_cols)

        for idx, col in enumerate(selected_cols):
            r, c = divmod(idx, n_cols)
            ax = axes[r, c]
            if groupby is not None:
                # Mimics EDA notebook: violin by group (e.g., stress_level)
                sns.violinplot(
                    data=df,
                    x=groupby,
                    y=col,
                    inner='box',
                    ax=ax,
                    palette='Set2'
                )
                ax.set_title(f'{col} by {groupby}', fontsize=13)
            else:
                # Violinplot for numeric feature (no x)
                sns.violinplot(
                    data=df,
                    y=col,
                    inner='box',
                    ax=ax,
                    palette='Set2'
                )
                ax.set_title(f'{col} Distribution', fontsize=13)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
        # Hide unused axes (mimics notebook)
        for j in range(idx + 1, n_rows * n_cols):
            r, c = divmod(j, n_cols)
            axes[r, c].axis('off')

        plt.tight_layout()
        fig.suptitle(title, fontsize=18, y=1.03)
        plt.savefig('results/violinplots_by_outlier_or_group.png', dpi=300, bbox_inches='tight')
        print("✓ Outlier/group violinplots saved to 'results/violinplots_by_outlier_or_group.png'")
        plt.show()
        
    def plot_feature_importance(self, feature_importance_df, top_n=20, title="Feature Importance"):
        """
        Plot feature importance
        
        Args:
            feature_importance_df: DataFrame with features and importance scores
            top_n (int): Number of top features to display
            title (str): Plot title
        """
        plt.figure(figsize=(12, 8))
        
        # Get top N features
        top_features = feature_importance_df.head(top_n)
        
        # Create horizontal bar plot
        ax = plt.barh(range(len(top_features)), top_features['Importance'], color='forestgreen', edgecolor='black')
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(top_features['Importance']):
            plt.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
        print("✓ Feature importance plot saved to 'results/feature_importance.png'")
        plt.show()
        
    def plot_model_comparison(self, comparison_df, metric='Accuracy', title="Model Comparison"):
        """
        Plot model comparison
        
        Args:
            comparison_df: DataFrame with model comparison results
            metric (str): Metric to compare
            title (str): Plot title
        """
        plt.figure(figsize=(12, 6))
        
        # Convert metric values to float
        values = comparison_df[metric].astype(float)
        models = comparison_df['Model']
        
        # Create bar plot
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        bars = plt.bar(range(len(models)), values, color=colors, edgecolor='black')
        
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        plt.ylabel(metric, fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylim(0, 1.1)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Model comparison plot saved to 'results/model_comparison.png'")
        plt.show()
        
    def plot_clustering_results(self, X, labels, title="Clustering Results (PCA 2D)"):
        """
        Plot clustering results using PCA for dimensionality reduction
        
        Args:
            X: Features
            labels: Cluster labels
            title (str): Plot title
        """
        plt.figure(figsize=(12, 8))
        
        # Apply PCA for 2D visualization
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        
        # Create scatter plot
        unique_labels = sorted(list(set(labels)))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            if label == -1:
                # Noise points (for DBSCAN)
                plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                          c='black', marker='x', s=50, alpha=0.5, label='Noise')
            else:
                plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                          c=[color], s=100, alpha=0.6, edgecolors='black',
                          label=f'Cluster {label}')
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/clustering_results.png', dpi=300, bbox_inches='tight')
        print("✓ Clustering results plot saved to 'results/clustering_results.png'")
        plt.show()
        
    def plot_elbow_curve(self, k_values, inertias, title="Elbow Method for Optimal K"):
        """
        Plot elbow curve for K-Means
        
        Args:
            k_values: List of k values
            inertias: List of inertia values
            title (str): Plot title
        """
        plt.figure(figsize=(10, 6))
        
        plt.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Inertia', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(k_values)
        
        plt.tight_layout()
        plt.savefig('results/elbow_curve.png', dpi=300, bbox_inches='tight')
        print("✓ Elbow curve saved to 'results/elbow_curve.png'")
        plt.show()
        
    def plot_silhouette_scores(self, k_values, silhouette_scores, title="Silhouette Score vs K"):
        """
        Plot silhouette scores for different k values
        
        Args:
            k_values: List of k values
            silhouette_scores: List of silhouette scores
            title (str): Plot title
        """
        plt.figure(figsize=(10, 6))
        
        plt.plot(k_values, silhouette_scores, 'go-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(k_values)
        
        # Mark the best k
        best_k_idx = np.argmax(silhouette_scores)
        best_k = k_values[best_k_idx]
        best_score = silhouette_scores[best_k_idx]
        plt.plot(best_k, best_score, 'r*', markersize=20, label=f'Best k={best_k}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('results/silhouette_scores.png', dpi=300, bbox_inches='tight')
        print("✓ Silhouette scores plot saved to 'results/silhouette_scores.png'")
        plt.show()
        
    def plot_data_distribution(self, data, columns=None, title="Data Distribution"):
        """
        Plot enhanced distributions of multiple numeric features (with KDE, mean and median lines).

        Args:
            data: DataFrame
            columns: List of columns to plot (if None, select key numerics, fallback to all numerics)
            title (str): Plot title
        """
        # Select columns as in EDA notebook, but fallback to all numerics if none found
        if columns is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            preferred = [
                'bullying', 'future_career_concerns', 'anxiety_level', 'depression', 'headache',
                'extracurricular_activities', 'peer_pressure', 'noise_level', 'mental_health_history'
            ]
            columns = [c for c in numeric_cols if c in preferred]
            if len(columns) == 0:
                columns = numeric_cols[:6]  # fallback to first six numeric columns

        n = len(columns)
        n_cols = 2  # reduced for better screen fit
        n_rows = int(np.ceil(n / n_cols)) if n > 0 else 0

        if n > 0:
            # Reduce overall figure height for screen fit; shrink subplot size
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(9.5, 3.5 * n_rows), dpi=120)
            axes = np.array(axes).reshape(n_rows, n_cols)
            for idx, col in enumerate(columns):
                r, c = divmod(idx, n_cols)
                ax = axes[r, c]
                sns.histplot(data[col], kde=True, ax=ax, color='skyblue', edgecolor='black')
                mean_val = data[col].mean()
                median_val = data[col].median()
                ax.axvline(mean_val, color='darkgreen', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='darkorange', linestyle='--', linewidth=1.5, label=f'Median: {median_val:.2f}')
                ax.legend(fontsize=8, loc="best", handlelength=1.15, frameon=False)
                ax.set_title(f'{col} distribution', fontsize=11)
                ax.set_xlabel(None)
                ax.set_ylabel(None)
            # Hide unused axes
            for j in range(idx + 1, n_rows * n_cols):
                r, c = divmod(j, n_cols)
                axes[r, c].axis('off')
            plt.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig('results/data_distribution.png', dpi=300, bbox_inches='tight')
            print("✓ Data distribution plot saved to 'results/data_distribution.png'")
            plt.show()
        else:
            print("No numeric columns to plot.")
        
    def create_results_folder(self):
        """Create results folder if it doesn't exist"""
        import os
        if not os.path.exists('results'):
            os.makedirs('results')
            print("✓ Created 'results' folder")
