"""
Data Preprocessing Module
Handles data loading, cleaning, transformation, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import os


class DataPreprocessor:
    """Class to handle all data preprocessing tasks"""
    
    def __init__(self, data_path):
        """
        Initialize the preprocessor
        
        Args:
            data_path (str): Path to the dataset file
        """
        self.data_path = data_path
        self.data = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """Load the dataset from CSV file"""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"✓ Data loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            return self.data
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return None
    
    def explore_data(self):
        """Perform basic data exploration"""
        if self.data is None:
            print("✗ No data loaded. Please load data first.")
            return
        
        print("\n" + "="*70)
        print("DATA EXPLORATION")
        print("="*70)
        
        print("\n1. Dataset Shape:")
        print(f"   Rows: {self.data.shape[0]}, Columns: {self.data.shape[1]}")
        
        print("\n2. Column Names and Types:")
        print(self.data.dtypes)
        
        print("\n3. First 5 Rows:")
        print(self.data.head())
        
        print("\n4. Statistical Summary:")
        print(self.data.describe())
        
        print("\n5. Missing Values:")
        missing = self.data.isnull().sum()
        if missing.sum() == 0:
            print("   No missing values found ✓")
        else:
            print(missing[missing > 0])
        
        print("\n6. Target Variable Distribution:")
        target_col = self.data.columns[-1]
        print(self.data[target_col].value_counts())
        
        return self.data.info()
    
    def handle_missing_values(self, strategy='mean'):
        """
        Handle missing values in the dataset
        
        Args:
            strategy (str): Strategy for imputation ('mean', 'median', 'most_frequent')
        """
        if self.data is None:
            print("✗ No data loaded.")
            return
        
        missing_count = self.data.isnull().sum().sum()
        
        if missing_count == 0:
            print("✓ No missing values to handle")
            return self.data
        
        print(f"\nHandling {missing_count} missing values using '{strategy}' strategy...")
        
        # Separate numeric and categorical columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        # Impute numeric columns
        if len(numeric_cols) > 0:
            imputer = SimpleImputer(strategy=strategy)
            self.data[numeric_cols] = imputer.fit_transform(self.data[numeric_cols])
        
        # Impute categorical columns
        if len(categorical_cols) > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            self.data[categorical_cols] = imputer.fit_transform(self.data[categorical_cols])
        
        print("✓ Missing values handled successfully")
        return self.data
    
    def remove_duplicates(self):
        """Remove duplicate rows from the dataset"""
        if self.data is None:
            print("✗ No data loaded.")
            return
        
        before = len(self.data)
        self.data = self.data.drop_duplicates()
        after = len(self.data)
        removed = before - after
        
        if removed > 0:
            print(f"✓ Removed {removed} duplicate rows")
        else:
            print("✓ No duplicate rows found")
        
        return self.data
    
    def encode_categorical(self, columns=None):
        """
        Encode categorical variables
        
        Args:
            columns (list): List of column names to encode. If None, encode all object columns
        """
        if self.data is None:
            print("✗ No data loaded.")
            return
        
        if columns is None:
            columns = self.data.select_dtypes(include=['object']).columns.tolist()
        
        if len(columns) == 0:
            print("✓ No categorical columns to encode")
            return self.data
        
        print(f"\nEncoding categorical columns: {columns}")
        
        for col in columns:
            if col in self.data.columns:
                self.data[col] = self.label_encoder.fit_transform(self.data[col].astype(str))
        
        print("✓ Categorical encoding completed")
        return self.data
    
    def prepare_features_target(self, target_col=None):
        """
        Prepare features (X) and target (y) for modeling
        
        Args:
            target_col (str): Name of the target column. If None, use last column
        
        Returns:
            tuple: (X, y) features and target
        """
        if self.data is None:
            print("✗ No data loaded.")
            return None, None
        
        if target_col is None:
            target_col = self.data.columns[-1]
        
        X = self.data.drop(columns=[target_col])
        y = self.data[target_col]
        
        print(f"\n✓ Features prepared: {X.shape[1]} features, {len(y)} samples")
        print(f"  Target column: '{target_col}'")
        print(f"  Target distribution:\n{y.value_counts()}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Args:
            X: Features
            y: Target
            test_size (float): Proportion of test set
            random_state (int): Random seed for reproducibility
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n✓ Data split completed:")
        print(f"  Training set: {len(X_train)} samples ({(1-test_size)*100:.0f}%)")
        print(f"  Testing set: {len(X_test)} samples ({test_size*100:.0f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, target_col=None, normalize=True, remove_dup=True):
        """
        Complete preprocessing pipeline
        
        Args:
            target_col (str): Name of target column
            normalize (bool): Whether to normalize features
            remove_dup (bool): Whether to remove duplicates
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("\n" + "="*70)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("="*70)
        
        self.load_data()
        self.explore_data()
        
        if remove_dup:
            self.remove_duplicates()
        
        self.handle_missing_values()
        
        if target_col is None:
            target_col = self.data.columns[-1]
        
        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        if target_col in categorical_cols:
            self.data[target_col] = self.label_encoder.fit_transform(self.data[target_col])
            categorical_cols.remove(target_col)
        
        if categorical_cols:
            self.encode_categorical(categorical_cols)
        
        X, y = self.prepare_features_target(target_col)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        if normalize:
            print("\nNormalizing features (fitting on training data only)...")
            X_train = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            print("✓ Normalization completed")
        
        print("\n" + "="*70)
        print("PREPROCESSING PIPELINE COMPLETED")
        print("="*70)
        
        return X_train, X_test, y_train, y_test
