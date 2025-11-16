"""
Student Stress Analysis - Main Application
A comprehensive Data Mining project analyzing factors leading to student stress
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import DataPreprocessor
from modeling import StressClassifier
from clustering import StressClusterer
from visualization import DataVisualizer
from utils import *
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

def main_menu():
    """Display main menu"""
    print_header("STUDENT STRESS ANALYSIS SYSTEM", width=80, char='=')
    print("\n📊 Data Mining Project: Main Factors Leading to Negative Stress in Students")
    print("   Dataset: Student Stress Monitoring Dataset")
    print("   Source: Kaggle - https://www.kaggle.com/datasets/mdsultanulislamovi/student-stress-monitoring-datasets")
    
    options = [
        "Data Preprocessing & Exploration",
        "Classification Analysis (Supervised Learning)",
        "Clustering Analysis (Unsupervised Learning)",
        "Visualizations & Reports",
        "Complete Analysis Pipeline",
        "Exit"
    ]
    
    print_menu(options)
    return get_user_choice(1, len(options))


def remove_outliers_iqr(df, columns):
    """Remove outliers using IQR method"""
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


def apply_pca(X, n_components=0.95):
    """Apply PCA for dimensionality reduction"""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    print(f"PCA reduced features to {X_pca.shape[1]} components, explaining {sum(pca.explained_variance_ratio_):.2%} variance")
    return X_pca, pca


def apply_rfe(X, y, n_features=10):
    """Apply Recursive Feature Elimination"""
    estimator = RandomForestClassifier()
    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
    X_rfe = rfe.fit_transform(X, y)
    selected_features = X.columns[rfe.support_].tolist()
    print(f"RFE selected {len(selected_features)} features: {selected_features}")
    return X_rfe, selected_features


def apply_smote(X, y):
    """Apply SMOTE for class imbalance if needed"""
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"SMOTE applied: Original shape {X.shape}, Resampled {X_res.shape}")
    return X_res, y_res


def preprocessing_menu():
    """Data preprocessing menu"""
    print_header("DATA PREPROCESSING & EXPLORATION", width=80)
    
    print("\nAvailable Datasets:")
    print("1. StressLevelDataset.csv (Numeric features)")
    print("2. Stress_Dataset.csv (Mixed features)")
    
    dataset_choice = get_user_choice(1, 2)
    
    if dataset_choice == 1:
        data_path = 'data/raw/StressLevelDataset.csv'
    else:
        data_path = 'data/raw/Stress_Dataset.csv'
    
    preprocessor = DataPreprocessor(data_path)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    print("\nRemoving outliers using IQR...")
    train_df = remove_outliers_iqr(train_df, numeric_cols)
    test_df = remove_outliers_iqr(test_df, numeric_cols)
    
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    
    print(f"Rows after outlier removal: Train {X_train.shape[0]}, Test {X_test.shape[0]}")
    
    print("\nClass distribution:")
    print(y_train.value_counts(normalize=True))
    if input("Apply SMOTE? (y/n): ").lower() == 'y':
        X_train, y_train = apply_smote(X_train, y_train)
    
    print("\nSaving processed data...")
    export_to_csv(train_df, 'train_data.csv', 'data/processed')
    export_to_csv(test_df, 'test_data.csv', 'data/processed')
    
    visualizer = DataVisualizer()
    visualizer.create_results_folder()
    visualizer.plot_target_distribution(y_train, "Training Set - Stress Level Distribution")
    
    pause()
    return X_train, X_test, y_train, y_test


def classification_menu(X_train=None, X_test=None, y_train=None, y_test=None):
    """Classification analysis menu"""
    print_header("CLASSIFICATION ANALYSIS", width=80)
    
    if X_train is None:
        print("\nLoading preprocessed data...")
        try:
            train_data = pd.read_csv('data/processed/train_data.csv')
            test_data = pd.read_csv('data/processed/test_data.csv')
            
            X_train = train_data.iloc[:, :-1]
            y_train = train_data.iloc[:, -1]
            X_test = test_data.iloc[:, :-1]
            y_test = test_data.iloc[:, -1]
            
            print("✓ Data loaded successfully")
        except FileNotFoundError:
            print("✗ Preprocessed data not found. Please run preprocessing first.")
            pause()
            return
    
    print("\nApplying advanced techniques...")
    X_train_pca, pca = apply_pca(X_train)
    X_test_pca = pca.transform(X_test)
    
    X_train_rfe, selected_features = apply_rfe(pd.DataFrame(X_train_pca), y_train, n_features=10)
    selected_indices = [int(f) if isinstance(f, (str, float)) else f for f in selected_features]
    X_test_rfe = X_test_pca[:, selected_indices]
    
    X_train_use = pd.DataFrame(X_train_rfe)
    X_test_use = pd.DataFrame(X_test_rfe)
    
    classifier = StressClassifier()
    
    print("\nClassification Options:")
    print("1. Train all models and compare")
    print("2. Train specific model")
    print("3. Get feature importance")
    
    choice = get_user_choice(1, 3)
    
    if choice == 1:
        classifier.train_all_models(X_train_use, y_train, X_test_use, y_test)
        comparison_df = classifier.compare_models()
        best_name, best_model, best_metrics = classifier.get_best_model()
        
        visualizer = DataVisualizer()
        visualizer.create_results_folder()
        
        y_pred = classifier.predict(best_name, X_test_use)
        visualizer.plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix - {best_name}")
        visualizer.plot_model_comparison(comparison_df, metric='Accuracy')
        
        save_results({
            'best_model': best_name,
            'accuracy': float(best_metrics['accuracy']),
            'precision': float(best_metrics['precision']),
            'recall': float(best_metrics['recall']),
            'f1_score': float(best_metrics['f1_score'])
        }, 'classification_results.json')
        
    elif choice == 2:
        # Train specific model
        print("\nAvailable Models:")
        models = list(classifier.models.keys())
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
        
        model_choice = get_user_choice(1, len(models))
        model_name = models[model_choice - 1]
        
        classifier.train_model(model_name, X_train_use, y_train)
        classifier.evaluate_model(model_name, X_test_use, y_test)
        classifier.print_evaluation(model_name)
        
        visualizer = DataVisualizer()
        visualizer.create_results_folder()
        y_pred = classifier.predict(model_name, X_test_use)
        visualizer.plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix - {model_name}")
        
    elif choice == 3:
        print("\nSelect model for feature importance:")
        tree_models = ['Random Forest', 'Decision Tree', 'Gradient Boosting']
        for i, model in enumerate(tree_models, 1):
            print(f"{i}. {model}")
        
        model_choice = get_user_choice(1, len(tree_models))
        model_name = tree_models[model_choice - 1]
        
        if model_name not in classifier.trained_models:
            classifier.train_model(model_name, X_train_use, y_train)
        
        feature_importance = classifier.get_feature_importance(model_name, selected_features)
        
        visualizer = DataVisualizer()
        visualizer.create_results_folder()
        visualizer.plot_feature_importance(feature_importance, top_n=15, 
                                          title=f"Top 15 Important Features - {model_name}")
    
    pause()


def clustering_menu(X_train=None):
    """Clustering analysis menu"""
    print_header("CLUSTERING ANALYSIS", width=80)
    
    if X_train is None:
        print("\nLoading preprocessed data...")
        try:
            train_data = pd.read_csv('data/processed/train_data.csv')
            X_train = train_data.iloc[:, :-1]
            print("✓ Data loaded successfully")
        except FileNotFoundError:
            print("✗ Preprocessed data not found. Please run preprocessing first.")
            pause()
            return
    
    X_train_pca, _ = apply_pca(X_train)
    X_train = pd.DataFrame(X_train_pca)
    
    clusterer = StressClusterer()
    visualizer = DataVisualizer()
    visualizer.create_results_folder()
    
    print("\nClustering Options:")
    print("1. Find optimal number of clusters")
    print("2. K-Means Clustering")
    print("3. Hierarchical Clustering")
    print("4. DBSCAN Clustering")
    print("5. Compare all clustering methods")
    
    choice = get_user_choice(1, 5)
    
    if choice == 1:
        results_df = clusterer.find_optimal_k(X_train, k_range=range(2, 11))
        visualizer.plot_elbow_curve(results_df['k'].tolist(), results_df['inertia'].tolist())
        visualizer.plot_silhouette_scores(results_df['k'].tolist(), results_df['silhouette'].tolist())
        
    elif choice == 2:
        k = int(input("\nEnter number of clusters (default=3): ") or "3")
        labels = clusterer.kmeans_clustering(X_train, n_clusters=k)
        clusterer.analyze_clusters(X_train, labels, [f'PC{i+1}' for i in range(X_train.shape[1])])
        visualizer.plot_clustering_results(X_train, labels, title=f"K-Means Clustering (k={k})")
        
    elif choice == 3:
        k = int(input("\nEnter number of clusters (default=3): ") or "3")
        labels = clusterer.hierarchical_clustering(X_train, n_clusters=k)
        clusterer.analyze_clusters(X_train, labels, [f'PC{i+1}' for i in range(X_train.shape[1])])
        visualizer.plot_clustering_results(X_train, labels, title=f"Hierarchical Clustering (k={k})")
        
    elif choice == 4:
        eps = float(input("\nEnter eps value (default=0.5): ") or "0.5")
        min_samples = int(input("Enter min_samples (default=5): ") or "5")
        labels = clusterer.dbscan_clustering(X_train, eps=eps, min_samples=min_samples)
        clusterer.analyze_clusters(X_train, labels, [f'PC{i+1}' for i in range(X_train.shape[1])])
        visualizer.plot_clustering_results(X_train, labels, title="DBSCAN Clustering")
        
    elif choice == 5:
        print("\nRunning all clustering methods...")
        clusterer.kmeans_clustering(X_train, n_clusters=3)
        clusterer.hierarchical_clustering(X_train, n_clusters=3)
        clusterer.dbscan_clustering(X_train, eps=0.5, min_samples=5)
        comparison_df = clusterer.compare_clustering_methods()
        export_to_csv(comparison_df, 'clustering_comparison.csv')
    
    pause()


def visualization_menu():
    """Visualization menu"""
    print_header("VISUALIZATIONS & REPORTS", width=80)
    
    print("\nVisualization Options:")
    print("1. View all generated plots")
    print("2. Generate correlation matrix")
    print("3. Generate data distribution plots (histograms/boxplots/violins)")
    print("4. Generate high correlation scatterplots")
    print("5. Generate outlier boxplots")
    print("6. Export results summary")
    
    choice = get_user_choice(1, 6)
    
    visualizer = DataVisualizer()
    visualizer.create_results_folder()
    
    if choice == 1:
        print("\n✓ All plots are saved in the 'results' folder")
        print("  Check the following files:")
        results_files = [
            "target_distribution.png",
            "correlation_matrix.png",
            "confusion_matrix.png",
            "feature_importance.png",
            "model_comparison.png",
            "clustering_results.png",
            "elbow_curve.png",
            "silhouette_scores.png",
            "data_distributions.png",
            "high_corr_scatterplots.png",
            "outlier_boxplots.png"
        ]
        for file in results_files:
            filepath = os.path.join('results', file)
            if os.path.exists(filepath):
                print(f"  ✓ {file}")
            else:
                print(f"  ✗ {file} (not generated yet)")
    
    elif choice == 2:
        try:
            train_data = pd.read_csv('data/processed/train_data.csv')
            visualizer.plot_correlation_matrix(train_data.iloc[:, :-1])
        except FileNotFoundError:
            print("✗ Preprocessed data not found. Please run preprocessing first.")
    
    elif choice == 3:
        try:
            train_data = pd.read_csv('data/processed/train_data.csv')
            X = train_data.iloc[:, :-1]
            visualizer.plot_data_distribution(X)
        except FileNotFoundError:
            print("✗ Preprocessed data not found. Please run preprocessing first.")
    
    elif choice == 4:
        try:
            train_data = pd.read_csv('data/processed/train_data.csv')
            X = train_data.iloc[:, :-1]
            cor_matrix = X.corr()
            upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
            corr_threshold = 0.7
            high_corr_pairs = [
                (row, col, cor_matrix.loc[row, col])
                for row in upper_tri.index
                for col in upper_tri.columns
                if pd.notnull(upper_tri.loc[row, col]) and abs(upper_tri.loc[row, col]) >= corr_threshold
            ]
            if high_corr_pairs:
                nplots = len(high_corr_pairs)
                rows = int(np.ceil(nplots / 2))
                cols = 2 if nplots > 1 else 1
                fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
                for idx, (x, y, corr_val) in enumerate(high_corr_pairs):
                    r, c = divmod(idx, cols)
                    ax = axes[r, c] if rows > 1 else axes[c] if cols > 1 else axes
                    sns.scatterplot(data=X, x=x, y=y, ax=ax, alpha=0.5)
                    sns.regplot(data=X, x=x, y=y, ax=ax, scatter=False, line_kws={'color': 'red', 'linewidth': 2})
                    ax.set_title(rf"$r$ = {corr_val:.2f}")
                plt.tight_layout()
                plt.savefig(os.path.join('results', 'high_corr_scatterplots.png'))
                plt.show()
            else:
                print("No high correlations found.")
        except FileNotFoundError:
            print("✗ Preprocessed data not found.")
    
    elif choice == 5:
        try:
            train_data = pd.read_csv('data/processed/train_data.csv')
            X = train_data.iloc[:, :-1]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(data=X, ax=ax)
            plt.xticks(rotation=90)
            plt.title("Boxplots for Outlier Detection")
            plt.savefig(os.path.join('results', 'outlier_boxplots.png'))
            plt.show()
        except FileNotFoundError:
            print("✗ Preprocessed data not found.")
    
    elif choice == 6:
        print("\nGenerating results summary...")
        report = []
        report.append("="*80)
        report.append("STUDENT STRESS ANALYSIS - RESULTS SUMMARY")
        report.append("="*80)
        report.append(f"\nGenerated on: {get_timestamp()}")
        report.append("\n" + "-"*80)
        
        try:
            class_results = load_results('classification_results.json')
            if class_results:
                report.append("\nCLASSIFICATION RESULTS:")
                report.append(f"  Best Model: {class_results.get('best_model', 'N/A')}")
                report.append(f"  Accuracy: {class_results.get('accuracy', 0):.4f}")
                report.append(f"  Precision: {class_results.get('precision', 0):.4f}")
                report.append(f"  Recall: {class_results.get('recall', 0):.4f}")
                report.append(f"  F1-Score: {class_results.get('f1_score', 0):.4f}")
        except:
            report.append("\nCLASSIFICATION RESULTS: Not available")
        
        report.append("\n" + "-"*80)
        report.append("\nFor detailed results, check the 'results' folder.")
        report.append("="*80)
        
        report_text = "\n".join(report)
        with open('results/summary_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print("\n✓ Summary report saved to 'results/summary_report.txt'")
    
    pause()


def complete_pipeline():
    """Run complete analysis pipeline"""
    print_header("COMPLETE ANALYSIS PIPELINE", width=80)
    print("\nThis will run the entire analysis pipeline:")
    print("1. Data Preprocessing (with outlier removal)")
    print("2. Advanced Techniques (PCA, RFE)")
    print("3. Classification Analysis")
    print("4. Clustering Analysis")
    print("5. Generate Visualizations (enhanced)")
    
    confirm = input("\nProceed? (y/n): ").lower()
    if confirm != 'y':
        return
    
    print_section("STEP 1: DATA PREPROCESSING")
    data_path = 'data/raw/StressLevelDataset.csv'
    preprocessor = DataPreprocessor(data_path)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    train_df = remove_outliers_iqr(train_df, numeric_cols)
    test_df = remove_outliers_iqr(test_df, numeric_cols)
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    
    if input("Apply SMOTE in pipeline? (y/n): ").lower() == 'y':
        X_train, y_train = apply_smote(X_train, y_train)
    
    export_to_csv(train_df, 'train_data.csv', 'data/processed')
    export_to_csv(test_df, 'test_data.csv', 'data/processed')
    
    print_section("STEP 2: ADVANCED TECHNIQUES")
    X_train_pca, pca = apply_pca(X_train)
    X_test_pca = pca.transform(X_test)
    X_train_rfe, selected_features = apply_rfe(pd.DataFrame(X_train_pca), y_train, n_features=10)
    X_test_rfe = X_test_pca[:, :len(selected_features)]
    X_train_use = pd.DataFrame(X_train_rfe)
    X_test_use = pd.DataFrame(X_test_rfe)
    
    print_section("STEP 3: CLASSIFICATION ANALYSIS")
    classifier = StressClassifier()
    classifier.train_all_models(X_train_use, y_train, X_test_use, y_test)
    comparison_df = classifier.compare_models()
    best_name, best_model, best_metrics = classifier.get_best_model()
    
    print_section("STEP 4: CLUSTERING ANALYSIS")
    clusterer = StressClusterer()
    clusterer.kmeans_clustering(pd.DataFrame(X_train_pca), n_clusters=3)
    clusterer.hierarchical_clustering(pd.DataFrame(X_train_pca), n_clusters=3)
    clusterer.compare_clustering_methods()
    
    print_section("STEP 5: GENERATING VISUALIZATIONS")
    visualizer = DataVisualizer()
    visualizer.create_results_folder()
    
    y_pred = classifier.predict(best_name, X_test_use)
    visualizer.plot_target_distribution(y_train)
    visualizer.plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix - {best_name}")
    visualizer.plot_model_comparison(comparison_df)
    
    feature_importance = classifier.get_feature_importance(best_name, selected_features)
    if feature_importance is not None:
        visualizer.plot_feature_importance(feature_importance, top_n=15)
    
    kmeans_labels = clusterer.get_cluster_labels('KMeans')
    visualizer.plot_clustering_results(pd.DataFrame(X_train_pca), kmeans_labels, title="K-Means Clustering Results")
    
    visualizer.plot_correlation_matrix(X_train)
    visualizer.plot_data_distribution(X_train)
    
    save_results({
        'best_model': best_name,
        'accuracy': float(best_metrics['accuracy']),
        'precision': float(best_metrics['precision']),
        'recall': float(best_metrics['recall']),
        'f1_score': float(best_metrics['f1_score']),
        'timestamp': get_timestamp()
    }, 'final_results.json')
    
    print_header("PIPELINE COMPLETED SUCCESSFULLY!", width=80)
    print("\n✓ All results saved to 'results' folder")
    print("✓ Processed data saved to 'data/processed' folder")
    
    pause()


def main():
    """Main application loop"""
    create_project_structure()
    
    X_train = X_test = y_train = y_test = None
    
    while True:
        choice = main_menu()
        
        if choice is None:
            break
        elif choice == 1:
            X_train, X_test, y_train, y_test = preprocessing_menu()
        elif choice == 2:
            classification_menu(X_train, X_test, y_train, y_test)
        elif choice == 3:
            clustering_menu(X_train)
        elif choice == 4:
            visualization_menu()
        elif choice == 5:
            complete_pipeline()
        elif choice == 6:
            print("\n👋 Thank you for using Student Stress Analysis System!")
            print("   Project by: Data Mining Team")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Program interrupted by user")
    except Exception as e:
        print(f"\n✗ An error occurred: {e}")
        import traceback
        traceback.print_exc()
