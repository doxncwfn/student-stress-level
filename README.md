# Student Stress Analysis - Data Mining Project

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Algorithms Implemented](#algorithms-implemented)
- [Results](#results)
- [Resources](#resources)
- [License](#license)

## Introduction

This project is a comprehensive **Data Mining** analysis focused on identifying and analyzing the **main factors leading to negative stress in students**. Using machine learning techniques, we explore student stress patterns through both supervised (classification) and unsupervised (clustering) learning approaches.

### Problem Statement
Student stress has become a significant concern in modern education. Understanding the factors that contribute to stress can help educational institutions develop better support systems and interventions. This project aims to:
- Identify key factors contributing to student stress
- Predict stress levels based on various features
- Discover patterns and groups among stressed students
- Provide actionable insights for stress management

## Project Overview

### Objectives
1. **Data Preprocessing**: Clean, transform, and prepare the dataset for analysis
2. **Exploratory Data Analysis**: Understand data distributions and relationships
3. **Classification Analysis**: Build predictive models to classify stress levels
4. **Clustering Analysis**: Discover natural groupings in student stress patterns
5. **Visualization**: Create comprehensive visualizations for insights
6. **Decision Making**: Draw conclusions and provide recommendations

### Key Technologies
- **Python 3.8+**
- **Scikit-learn**: Machine learning algorithms
- **Pandas & NumPy**: Data manipulation
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter**: Interactive analysis

## Dataset Information

### Source
**Kaggle Dataset**: [Student Stress Monitoring Datasets](https://www.kaggle.com/datasets/mdsultanulislamovi/student-stress-monitoring-datasets)

### Datasets Included
1. **StressLevelDataset.csv** (1,102 records, 21 features)
   - Numeric features measuring various stress factors
   - Target: `stress_level` (0: Low, 1: Moderate, 2: High)

2. **Stress_Dataset.csv** (844 records, 26 features)
   - Mixed features (numeric and categorical)
   - Target: Stress type classification

### Key Features
- **Psychological Factors**: Anxiety level, depression, self-esteem, mental health history
- **Physical Factors**: Headaches, blood pressure, sleep quality, breathing problems
- **Environmental Factors**: Noise level, living conditions, safety, basic needs
- **Academic Factors**: Academic performance, study load, teacher-student relationship
- **Social Factors**: Social support, peer pressure, bullying, extracurricular activities
- **Future Concerns**: Career concerns, future anxiety

## Features

### 1. Data Preprocessing
- Data loading and validation
- Missing value handling
- Duplicate removal
- Categorical encoding
- Feature normalization
- Train-test splitting

### 2. Classification Models
- Random Forest Classifier
- Decision Tree Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Logistic Regression
- Gradient Boosting

### 3. Clustering Algorithms
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN
- Optimal K selection (Elbow method, Silhouette score)

### 4. Visualizations
- Target distribution plots
- Correlation matrices
- Confusion matrices
- Feature importance charts
- Model comparison plots
- Clustering visualizations (PCA)
- Elbow curves
- Silhouette score plots

### 5. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Classification Report
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/data-mining-assignment.git
cd data-mining-assignment
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python --version
pip list
```

## Usage

### Running the Main Application

#### Interactive Menu System
```bash
python main.py
```

The application provides an interactive menu with the following options:
1. **Data Preprocessing & Exploration** - Load and preprocess the dataset
2. **Classification Analysis** - Train and evaluate classification models
3. **Clustering Analysis** - Perform clustering analysis
4. **Visualizations & Reports** - Generate plots and reports
5. **Complete Analysis Pipeline** - Run entire analysis automatically
6. **Exit** - Close the application

### Quick Start Example

#### Option 1: Complete Pipeline (Recommended for first run)
```bash
python main.py
# Select option 5: Complete Analysis Pipeline
```

#### Option 2: Step-by-Step Analysis
```bash
python main.py
# 1. Select option 1: Data Preprocessing
# 2. Select option 2: Classification Analysis
# 3. Select option 3: Clustering Analysis
# 4. Select option 4: View Visualizations
```

### Using Individual Modules

#### Example: Data Preprocessing
```python
from src.preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor('data/raw/StressLevelDataset.csv')

# Run preprocessing pipeline
X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
```

#### Example: Classification
```python
from src.modeling import StressClassifier

# Initialize classifier
classifier = StressClassifier()

# Train all models
classifier.train_all_models(X_train, y_train, X_test, y_test)

# Compare models
comparison = classifier.compare_models()

# Get best model
best_name, best_model, metrics = classifier.get_best_model()
```

#### Example: Clustering
```python
from src.clustering import StressClusterer

# Initialize clusterer
clusterer = StressClusterer()

# Perform K-Means clustering
labels = clusterer.kmeans_clustering(X_train, n_clusters=3)

# Analyze clusters
clusterer.analyze_clusters(X_train, labels)
```

## Project Structure

```
data-mining-assignment/
│
├── data/
│   ├── raw/                          # Original datasets
│   │   ├── StressLevelDataset.csv
│   │   └── Stress_Dataset.csv
│   └── processed/                    # Processed datasets
│       ├── train_data.csv
│       └── test_data.csv
│
├── src/                              # Source code modules
│   ├── preprocessing.py              # Data preprocessing
│   ├── modeling.py                   # Classification models
│   ├── clustering.py                 # Clustering algorithms
│   ├── visualization.py              # Visualization functions
│   └── utils.py                      # Utility functions
│
├── results/                          # Output files
│   ├── *.png                         # Generated plots
│   ├── *.csv                         # Result tables
│   ├── *.json                        # Metrics and results
│   └── summary_report.txt            # Final report
│
├── docs/                             # Documentation
│   ├── main.tex                      # LaTeX report
│   └── main.pdf                      # PDF report
│
├── main.py                           # Main application
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── LICENSE                           # License file
└── .gitignore                        # Git ignore rules
```

## Algorithms Implemented

### Classification Algorithms

#### 1. Random Forest Classifier
- **Type**: Ensemble learning method
- **Advantages**: High accuracy, handles non-linear relationships, feature importance
- **Use Case**: Best overall performance for stress prediction

#### 2. Decision Tree
- **Type**: Tree-based model
- **Advantages**: Interpretable, handles both numerical and categorical data
- **Use Case**: Understanding decision rules for stress classification

#### 3. Support Vector Machine (SVM)
- **Type**: Kernel-based classifier
- **Advantages**: Effective in high-dimensional spaces
- **Use Case**: Complex decision boundaries

#### 4. K-Nearest Neighbors (KNN)
- **Type**: Instance-based learning
- **Advantages**: Simple, no training phase
- **Use Case**: Pattern recognition in similar students

#### 5. Naive Bayes
- **Type**: Probabilistic classifier
- **Advantages**: Fast, works well with small datasets
- **Use Case**: Baseline model for comparison

#### 6. Logistic Regression
- **Type**: Linear model
- **Advantages**: Interpretable coefficients
- **Use Case**: Understanding linear relationships

#### 7. Gradient Boosting
- **Type**: Ensemble boosting method
- **Advantages**: High accuracy, handles complex patterns
- **Use Case**: Advanced predictive modeling

### Clustering Algorithms

#### 1. K-Means Clustering
- **Type**: Centroid-based clustering
- **Advantages**: Fast, scalable, easy to interpret
- **Use Case**: Identifying main stress groups

#### 2. Hierarchical Clustering
- **Type**: Tree-based clustering
- **Advantages**: No need to specify K, creates dendrogram
- **Use Case**: Understanding hierarchical stress patterns

#### 3. DBSCAN
- **Type**: Density-based clustering
- **Advantages**: Finds arbitrary shapes, identifies outliers
- **Use Case**: Detecting unusual stress patterns

## Results

### Classification Performance

Expected results (will vary based on data):

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | ~85-90% | ~0.85 | ~0.85 | ~0.85 |
| Gradient Boosting | ~83-88% | ~0.83 | ~0.83 | ~0.83 |
| Decision Tree | ~75-82% | ~0.75 | ~0.75 | ~0.75 |
| SVM | ~78-85% | ~0.78 | ~0.78 | ~0.78 |
| KNN | ~75-80% | ~0.75 | ~0.75 | ~0.75 |
| Logistic Regression | ~70-78% | ~0.70 | ~0.70 | ~0.70 |
| Naive Bayes | ~68-75% | ~0.68 | ~0.68 | ~0.68 |

### Key Findings

1. **Most Important Stress Factors**:
   - Anxiety level
   - Depression
   - Academic performance
   - Sleep quality
   - Future career concerns

2. **Stress Level Distribution**:
   - Low Stress: ~30-35%
   - Moderate Stress: ~40-45%
   - High Stress: ~20-25%

3. **Clustering Insights**:
   - Optimal number of clusters: 3-4
   - Clear separation between stress groups
   - Environmental and academic factors strongly correlate

## Resources

### Data Mining Concepts
1. **Data Preprocessing**
   - Data cleaning techniques
   - Feature scaling and normalization
   - Handling missing values
   - Encoding categorical variables

2. **Classification**
   - Supervised learning principles
   - Decision trees and ensemble methods
   - Support Vector Machines
   - Model evaluation metrics

3. **Clustering**
   - Unsupervised learning principles
   - Distance metrics
   - Cluster validation
   - Dimensionality reduction

### Recommended Reading
- **Books**:
  - "Data Mining: Concepts and Techniques" by Jiawei Han
  - "Pattern Recognition and Machine Learning" by Christopher Bishop
  - "Introduction to Data Mining" by Pang-Ning Tan

- **Online Courses**:
  - Coursera: Machine Learning by Andrew Ng
  - edX: Data Science and Machine Learning
  - Kaggle: Learn Data Science

- **Documentation**:
  - [Scikit-learn Documentation](https://scikit-learn.org/stable/)
  - [Pandas Documentation](https://pandas.pydata.org/docs/)
  - [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

### Useful Links
- [Kaggle Dataset](https://www.kaggle.com/datasets/mdsultanulislamovi/student-stress-monitoring-datasets)
- [Scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/index.html)
- [Data Mining Techniques](https://www.kdnuggets.com/)
- [Machine Learning Mastery](https://machinelearningmastery.com/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset provided by [Md Sultan Ul Islam Ovi](https://www.kaggle.com/mdsultanulislamovi) on Kaggle
- Course instructors and teaching assistants
- Open-source community for the amazing tools and libraries

---

Made with ❤️ for Data Mining Course

**Note**: This project is for educational purposes only. The findings and recommendations should be validated by domain experts before practical application.
