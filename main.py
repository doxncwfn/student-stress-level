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


def preprocessing_menu():
    """Data preprocessing menu"""
    print_header("DATA PREPROCESSING & EXPLORATION", width=80)
    
    # Choose dataset
    print("\nAvailable Datasets:")
    print("1. StressLevelDataset.csv (Numeric features)")
    print("2. Stress_Dataset.csv (Mixed features)")
    
    dataset_choice = get_user_choice(1, 2)
    
    if dataset_choice == 1:
        data_path = 'data/raw/StressLevelDataset.csv'
    else:
        data_path = 'data/raw/Stress_Dataset.csv'
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(data_path)
    
    # Run preprocessing pipeline
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
    
    # Save processed data
    print("\nSaving processed data...")
    export_to_csv(pd.concat([X_train, y_train], axis=1), 'train_data.csv', 'data/processed')
    export_to_csv(pd.concat([X_test, y_test], axis=1), 'test_data.csv', 'data/processed')
    
    # Visualize
    visualizer = DataVisualizer()
    visualizer.create_results_folder()
    visualizer.plot_target_distribution(y_train, "Training Set - Stress Level Distribution")
    
    pause()
    return X_train, X_test, y_train, y_test


def classification_menu(X_train=None, X_test=None, y_train=None, y_test=None):
    """Classification analysis menu"""
    print_header("CLASSIFICATION ANALYSIS", width=80)
    
    # Load data if not provided
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
    
    # Initialize classifier
    classifier = StressClassifier()
    
    print("\nClassification Options:")
    print("1. Train all models and compare")
    print("2. Train specific model")
    print("3. Get feature importance")
    
    choice = get_user_choice(1, 3)
    
    if choice == 1:
        # Train all models
        classifier.train_all_models(X_train, y_train, X_test, y_test)
        
        # Compare models
        comparison_df = classifier.compare_models()
        
        # Get best model
        best_name, best_model, best_metrics = classifier.get_best_model()
        
        # Visualize results
        visualizer = DataVisualizer()
        visualizer.create_results_folder()
        
        # Plot confusion matrix for best model
        y_pred = classifier.predict(best_name, X_test)
        visualizer.plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix - {best_name}")
        
        # Plot model comparison
        visualizer.plot_model_comparison(comparison_df, metric='Accuracy')
        
        # Save results
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
        
        classifier.train_model(model_name, X_train, y_train)
        classifier.evaluate_model(model_name, X_test, y_test)
        classifier.print_evaluation(model_name)
        
        # Visualize
        visualizer = DataVisualizer()
        visualizer.create_results_folder()
        y_pred = classifier.predict(model_name, X_test)
        visualizer.plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix - {model_name}")
        
    elif choice == 3:
        # Feature importance
        print("\nSelect model for feature importance:")
        tree_models = ['Random Forest', 'Decision Tree', 'Gradient Boosting']
        for i, model in enumerate(tree_models, 1):
            print(f"{i}. {model}")
        
        model_choice = get_user_choice(1, len(tree_models))
        model_name = tree_models[model_choice - 1]
        
        # Train if not already trained
        if model_name not in classifier.trained_models:
            classifier.train_model(model_name, X_train, y_train)
        
        # Get feature importance
        feature_importance = classifier.get_feature_importance(model_name, X_train.columns.tolist())
        
        # Visualize
        visualizer = DataVisualizer()
        visualizer.create_results_folder()
        visualizer.plot_feature_importance(feature_importance, top_n=15, 
                                          title=f"Top 15 Important Features - {model_name}")
    
    pause()


def clustering_menu(X_train=None):
    """Clustering analysis menu"""
    print_header("CLUSTERING ANALYSIS", width=80)
    
    # Load data if not provided
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
    
    # Initialize clusterer
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
        # Find optimal K
        results_df = clusterer.find_optimal_k(X_train, k_range=range(2, 11))
        
        # Visualize
        visualizer.plot_elbow_curve(results_df['k'].tolist(), results_df['inertia'].tolist())
        visualizer.plot_silhouette_scores(results_df['k'].tolist(), results_df['silhouette'].tolist())
        
    elif choice == 2:
        # K-Means
        k = int(input("\nEnter number of clusters (default=3): ") or "3")
        labels = clusterer.kmeans_clustering(X_train, n_clusters=k)
        
        # Analyze clusters
        clusterer.analyze_clusters(X_train, labels, X_train.columns.tolist())
        
        # Visualize
        visualizer.plot_clustering_results(X_train, labels, title=f"K-Means Clustering (k={k})")
        
    elif choice == 3:
        # Hierarchical
        k = int(input("\nEnter number of clusters (default=3): ") or "3")
        labels = clusterer.hierarchical_clustering(X_train, n_clusters=k)
        
        # Analyze clusters
        clusterer.analyze_clusters(X_train, labels, X_train.columns.tolist())
        
        # Visualize
        visualizer.plot_clustering_results(X_train, labels, title=f"Hierarchical Clustering (k={k})")
        
    elif choice == 4:
        # DBSCAN
        eps = float(input("\nEnter eps value (default=0.5): ") or "0.5")
        min_samples = int(input("Enter min_samples (default=5): ") or "5")
        labels = clusterer.dbscan_clustering(X_train, eps=eps, min_samples=min_samples)
        
        # Analyze clusters
        clusterer.analyze_clusters(X_train, labels, X_train.columns.tolist())
        
        # Visualize
        visualizer.plot_clustering_results(X_train, labels, title="DBSCAN Clustering")
        
    elif choice == 5:
        # Compare all methods
        print("\nRunning all clustering methods...")
        clusterer.kmeans_clustering(X_train, n_clusters=3)
        clusterer.hierarchical_clustering(X_train, n_clusters=3)
        clusterer.dbscan_clustering(X_train, eps=0.5, min_samples=5)
        
        # Compare
        comparison_df = clusterer.compare_clustering_methods()
        
        # Save results
        export_to_csv(comparison_df, 'clustering_comparison.csv')
    
    pause()


def visualization_menu():
    """Visualization menu"""
    print_header("VISUALIZATIONS & REPORTS", width=80)
    
    print("\nVisualization Options:")
    print("1. View all generated plots")
    print("2. Generate correlation matrix")
    print("3. Generate data distribution plots")
    print("4. Export results summary")
    
    choice = get_user_choice(1, 4)
    
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
            "silhouette_scores.png"
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
            visualizer = DataVisualizer()
            visualizer.create_results_folder()
            visualizer.plot_correlation_matrix(train_data.iloc[:, :-1])
        except FileNotFoundError:
            print("✗ Preprocessed data not found. Please run preprocessing first.")
    
    elif choice == 3:
        try:
            train_data = pd.read_csv('data/processed/train_data.csv')
            visualizer = DataVisualizer()
            visualizer.create_results_folder()
            visualizer.plot_data_distribution(train_data.iloc[:, :-1])
        except FileNotFoundError:
            print("✗ Preprocessed data not found. Please run preprocessing first.")
    
    elif choice == 4:
        print("\nGenerating results summary...")
        # Create a comprehensive report
        report = []
        report.append("="*80)
        report.append("STUDENT STRESS ANALYSIS - RESULTS SUMMARY")
        report.append("="*80)
        report.append(f"\nGenerated on: {get_timestamp()}")
        report.append("\n" + "-"*80)
        
        # Load classification results
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
        
        # Save report
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
    print("1. Data Preprocessing")
    print("2. Classification Analysis")
    print("3. Clustering Analysis")
    print("4. Generate Visualizations")
    
    confirm = input("\nProceed? (y/n): ").lower()
    if confirm != 'y':
        return
    
    # Step 1: Preprocessing
    print_section("STEP 1: DATA PREPROCESSING")
    dataset_choice = get_user_choice(1, 2)
    
    if dataset_choice == 1:
        data_path = 'data/raw/StressLevelDataset.csv'
    else:
        data_path = 'data/raw/Stress_Dataset.csv'
    preprocessor = DataPreprocessor(data_path)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
    
    # Save processed data
    export_to_csv(pd.concat([X_train, y_train], axis=1), 'train_data.csv', 'data/processed')
    export_to_csv(pd.concat([X_test, y_test], axis=1), 'test_data.csv', 'data/processed')
    
    # Step 2: Classification
    print_section("STEP 2: CLASSIFICATION ANALYSIS")
    classifier = StressClassifier()
    classifier.train_all_models(X_train, y_train, X_test, y_test)
    comparison_df = classifier.compare_models()
    best_name, best_model, best_metrics = classifier.get_best_model()
    
    # Step 3: Clustering
    print_section("STEP 3: CLUSTERING ANALYSIS")
    clusterer = StressClusterer()
    clusterer.kmeans_clustering(X_train, n_clusters=3)
    clusterer.hierarchical_clustering(X_train, n_clusters=3)
    clusterer.compare_clustering_methods()
    
    # Step 4: Visualizations
    print_section("STEP 4: GENERATING VISUALIZATIONS")
    visualizer = DataVisualizer()
    visualizer.create_results_folder()
    
    # Classification visualizations
    y_pred = classifier.predict(best_name, X_test)
    visualizer.plot_target_distribution(y_train)
    visualizer.plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix - {best_name}")
    visualizer.plot_model_comparison(comparison_df)
    
    # Feature importance
    feature_importance = classifier.get_feature_importance(best_name, X_train.columns.tolist())
    if feature_importance is not None:
        visualizer.plot_feature_importance(feature_importance, top_n=15)
    
    # Clustering visualizations
    kmeans_labels = clusterer.get_cluster_labels('KMeans')
    visualizer.plot_clustering_results(X_train, kmeans_labels, title="K-Means Clustering Results")
    
    # Correlation matrix
    visualizer.plot_correlation_matrix(X_train)
    
    # Save final results
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
    
    # Shared data variables
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
