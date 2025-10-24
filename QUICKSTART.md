# Quick Start Guide - Student Stress Analysis

## Get Started

### Step 1: Install Dependencies (1 minute)
```bash
pip install -r requirements.txt
```

### Step 2: Run the Application (30 seconds)
```bash
python main.py
```

### Step 3: Choose Option 5 - Complete Analysis Pipeline (3 minutes)
This will automatically:
- Load and preprocess the data
- Train all 7 classification models
- Perform clustering analysis
- Generate all visualizations
- Save results to `results/` folder

---

## What You'll Get

### Generated Files

#### In `data/processed/` folder:
- `train_data.csv` - Training dataset (80%)
- `test_data.csv` - Testing dataset (20%)

#### In `results/` folder:
- `target_distribution.png` - Stress level distribution
- `correlation_matrix.png` - Feature correlations
- `confusion_matrix.png` - Best model confusion matrix
- `feature_importance.png` - Top 15 important features
- `model_comparison.png` - All models comparison
- `clustering_results.png` - K-Means clustering visualization
- `classification_results.json` - Model metrics
- `final_results.json` - Complete results summary

---

## Alternative: Step-by-Step Approach

### Option 1: Data Preprocessing Only
```bash
python main.py
# Select: 1. Data Preprocessing & Exploration
# Choose dataset: 1 (StressLevelDataset.csv)
```

### Option 2: Classification Analysis
```bash
python main.py
# Select: 2. Classification Analysis
# Choose: 1. Train all models and compare
```

### Option 3: Clustering Analysis
```bash
python main.py
# Select: 3. Clustering Analysis
# Choose: 5. Compare all clustering methods
```

### Option 4: View Results
```bash
python main.py
# Select: 4. Visualizations & Reports
# Choose: 1. View all generated plots
```

---

## Using Python Code Directly

### Example 1: Quick Classification
```python
from src.preprocessing import DataPreprocessor
from src.modeling import StressClassifier

# Preprocess data
preprocessor = DataPreprocessor('data/raw/StressLevelDataset.csv')
X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()

# Train and evaluate
classifier = StressClassifier()
classifier.train_all_models(X_train, y_train, X_test, y_test)
classifier.compare_models()
best_name, best_model, metrics = classifier.get_best_model()

print(f"Best Model: {best_name}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### Example 2: Quick Clustering
```python
from src.preprocessing import DataPreprocessor
from src.clustering import StressClusterer
from src.visualization import DataVisualizer

# Preprocess data
preprocessor = DataPreprocessor('data/raw/StressLevelDataset.csv')
X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()

# Perform clustering
clusterer = StressClusterer()
labels = clusterer.kmeans_clustering(X_train, n_clusters=3)
clusterer.analyze_clusters(X_train, labels, X_train.columns.tolist())

# Visualize
visualizer = DataVisualizer()
visualizer.create_results_folder()
visualizer.plot_clustering_results(X_train, labels)
```

---

## Expected Results

### Classification Performance
- **Best Model:** Random Forest or Gradient Boosting
- **Expected Accuracy:** 85-90%
- **Training Time:** 2-3 minutes for all models

### Clustering Results
- **Optimal Clusters:** 3-4 groups
- **Main Patterns:** Low, Moderate, High stress groups
- **Silhouette Score:** 0.3-0.5 (good separation)

### Top Stress Factors
1. Anxiety level
2. Depression
3. Academic performance
4. Sleep quality
5. Future career concerns

---

## Troubleshooting

### Issue: Import Error
```bash
# Solution: Make sure you're in the project directory
cd data-mining-assignment
python main.py
```

### Issue: Module Not Found
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

### Issue: File Not Found
```bash
# Solution: Ensure datasets are in data/raw/ folder
# Check: data/raw/StressLevelDataset.csv exists
```

### Issue: Memory Error
```bash
# Solution: Use smaller dataset or reduce n_estimators
# Edit src/modeling.py: n_estimators=50 instead of 100
```

---

## Performance Tips

### For Faster Execution
1. Use StressLevelDataset.csv (smaller, numeric only)
2. Reduce number of models (comment out some in modeling.py)
3. Reduce n_estimators in Random Forest (line 23 in modeling.py)

### For Better Results
1. Perform hyperparameter tuning
2. Try feature selection
3. Use cross-validation
4. Experiment with different preprocessing strategies

---

## Next Steps

1. **Explore the Code**
   - Read through `src/` modules
   - Understand each function
   - Modify parameters

2. **Analyze Results**
   - Check `results/` folder
   - Review generated plots
   - Read JSON result files

3. **Customize**
   - Add new models
   - Create new visualizations
   - Implement feature engineering

4. **Document**
   - Write your findings
   - Create presentation
   - Prepare report

---

## Learning Path

### Beginner
1. Run complete pipeline (Option 5)
2. View generated visualizations
3. Read the results
4. Understand the workflow

### Intermediate
1. Run step-by-step (Options 1-4)
2. Modify parameters
3. Try different models
4. Analyze feature importance

### Advanced
1. Implement new algorithms
2. Perform hyperparameter tuning
3. Add feature engineering
4. Create custom visualizations
5. Write research paper

---

**Estimated Total Time:** 5-10 minutes for complete analysis

**Have fun!**
