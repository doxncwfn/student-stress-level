"""
Modeling Module
Implements various classification algorithms for stress level prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')


class StressClassifier:
    """Class to handle classification models for stress prediction"""
    
    def __init__(self):
        """Initialize classifiers"""
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'SVM': SVC(kernel='rbf', random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        self.trained_models = {}
        self.results = {}
        
    def train_model(self, model_name, X_train, y_train):
        """
        Train a specific model
        
        Args:
            model_name (str): Name of the model to train
            X_train: Training features
            y_train: Training target
        """
        if model_name not in self.models:
            print(f"✗ Model '{model_name}' not found!")
            return None
        
        print(f"\nTraining {model_name}...")
        model = self.models[model_name]
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        print(f"✓ {model_name} trained successfully")
        
        return model
    
    def predict(self, model_name, X_test):
        """
        Make predictions using a trained model
        
        Args:
            model_name (str): Name of the trained model
            X_test: Test features
        
        Returns:
            array: Predictions
        """
        if model_name not in self.trained_models:
            print(f"✗ Model '{model_name}' not trained yet!")
            return None
        
        return self.trained_models[model_name].predict(X_test)
    
    def evaluate_model(self, model_name, X_test, y_test):
        """
        Evaluate a trained model
        
        Args:
            model_name (str): Name of the model
            X_test: Test features
            y_test: Test target
        
        Returns:
            dict: Evaluation metrics
        """
        if model_name not in self.trained_models:
            print(f"✗ Model '{model_name}' not trained yet!")
            return None
        
        y_pred = self.predict(model_name, X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, zero_division=0)
        }
        
        self.results[model_name] = metrics
        
        return metrics
    
    def print_evaluation(self, model_name):
        """
        Print evaluation results for a model
        
        Args:
            model_name (str): Name of the model
        """
        if model_name not in self.results:
            print(f"✗ No evaluation results for '{model_name}'")
            return
        
        metrics = self.results[model_name]
        
        print(f"\n{'='*70}")
        print(f"EVALUATION RESULTS: {model_name}")
        print(f"{'='*70}")
        print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print(f"\nClassification Report:")
        print(metrics['classification_report'])
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate all models
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
        
        Returns:
            dict: Results for all models
        """
        print("\n" + "="*70)
        print("TRAINING ALL CLASSIFICATION MODELS")
        print("="*70)
        
        for model_name in self.models.keys():
            self.train_model(model_name, X_train, y_train)
            self.evaluate_model(model_name, X_test, y_test)
        
        print("\n✓ All models trained and evaluated")
        return self.results
    
    def compare_models(self):
        """
        Compare all trained models
        
        Returns:
            DataFrame: Comparison of model performances
        """
        if not self.results:
            print("✗ No models evaluated yet!")
            return None
        
        comparison = []
        for model_name, metrics in self.results.items():
            comparison.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}"
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('Accuracy', ascending=False)
        
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        print(df.to_string(index=False))
        
        return df
    
    def get_best_model(self):
        """
        Get the best performing model based on accuracy
        
        Returns:
            tuple: (model_name, model, metrics)
        """
        if not self.results:
            print("✗ No models evaluated yet!")
            return None, None, None
        
        best_model_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        best_model = self.trained_models[best_model_name]
        best_metrics = self.results[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   Accuracy: {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)")
        
        return best_model_name, best_model, best_metrics
    
    def get_feature_importance(self, model_name, feature_names):
        """
        Get feature importance for tree-based models
        
        Args:
            model_name (str): Name of the model
            feature_names (list): List of feature names
        
        Returns:
            DataFrame: Feature importance
        """
        if model_name not in self.trained_models:
            print(f"✗ Model '{model_name}' not trained yet!")
            return None
        
        model = self.trained_models[model_name]
        
        # Check if model has feature_importances_ attribute
        if not hasattr(model, 'feature_importances_'):
            print(f"✗ Model '{model_name}' does not support feature importance")
            return None
        
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print(f"\n{'='*70}")
        print(f"FEATURE IMPORTANCE: {model_name}")
        print(f"{'='*70}")
        print(feature_importance.to_string(index=False))
        
        return feature_importance
    
    def predict_stress_level(self, model_name, features):
        """
        Predict stress level for new data
        
        Args:
            model_name (str): Name of the trained model
            features: Feature values
        
        Returns:
            int: Predicted stress level
        """
        if model_name not in self.trained_models:
            print(f"✗ Model '{model_name}' not trained yet!")
            return None
        
        prediction = self.trained_models[model_name].predict([features])
        return prediction[0]
