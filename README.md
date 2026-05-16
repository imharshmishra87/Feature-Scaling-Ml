# 🔧 Feature Scaling & Preprocessing - Complete Data Preparation Guide

A comprehensive collection of machine learning preprocessing and feature engineering techniques with Jupyter notebooks covering data cleaning, scaling, transformations, and production-ready pipelines.

## 📋 Overview

This repository contains step-by-step implementations of essential data preprocessing techniques. Learn how to prepare raw data for machine learning models through feature scaling, handling missing values, detecting outliers, encoding categorical variables, and building production pipelines.

**Perfect for:** Data scientists preparing datasets, ML practitioners learning best practices, and anyone building robust ML pipelines.

---

## 🎯 What's Inside

### 📊 Feature Scaling Techniques

#### Standardization & Normalization
- **[feature_scaling(standardization).ipynb](feature_scaling(standardization).ipynb)** - Zero-mean, unit-variance scaling
  - StandardScaler theory and implementation
  - When to use standardization
  - Practical examples on real datasets
  - Comparison with other scaling methods

#### Power Transformation
- **[powertransformer.ipynb](powertransformer.ipynb)** - Box-Cox and Yeo-Johnson transformations
  - Normalize skewed distributions
  - Make data more Gaussian-like
  - Improve model performance
  - Implementation and comparison

---

### 🧹 Handling Missing Values

#### Simple Imputation Strategies
- **[handling_missing_values(_simple_imputer).ipynb](handling_missing_values(_simple_imputer).ipynb)** - Basic filling techniques
  - Mean imputation
  - Median imputation
  - Most frequent value
  - Forward fill and backward fill
  - When to use each strategy

#### Advanced Imputation Methods
- **[knn_imputation.ipynb](knn_imputation.ipynb)** - K-Nearest Neighbors imputation
  - Find similar samples
  - Fill with neighboring values
  - Preserve data relationships
  - Parameter tuning

- **[random_imputation.ipynb](random_imputation.ipynb)** - Random value imputation
  - Add random noise
  - Preserve distribution
  - Avoid bias from single values
  - Implementation tips

#### Missing Indicator Features
- **[Missing_indicator.ipynb](Missing_indicator.ipynb)** - Create binary indicators
  - Flag missing values
  - Capture missingness patterns
  - Combine with other methods
  - Domain-specific applications

---

### 🎯 Outlier Detection & Removal

#### Statistical Methods
- **[Outlier_detection(IQR).ipynb](Outlier_detection(IQR).ipynb)** - Interquartile Range method
  - Identify outliers using quartiles
  - Set detection thresholds
  - Handle extreme values
  - Robust to distributions

- **[outlier_detection(z_score).ipynb](outlier_detection(z_score).ipynb)** - Z-Score method
  - Measure deviation from mean
  - Statistical significance testing
  - Assumes normal distribution
  - Parameter selection

- **[outlier_detection(percentile_method).ipynb](outlier_detection(percentile_method).ipynb)** - Percentile approach
  - Flexible thresholding
  - Domain-specific cutoffs
  - Handling skewed data
  - Real-world applications

---

### 🔄 Data Transformation & Encoding

#### Categorical Variables
- **[Encoding.ipynb](Encoding.ipynb)** - Convert categorical to numerical
  - Label encoding
  - One-hot encoding
  - Ordinal encoding
  - Target encoding
  - When to use each method

#### Mixed Variables Handling
- **[handling_mixed_var.ipynb](handling_mixed_var.ipynb)** - Work with mixed data types
  - Numerical features preprocessing
  - Categorical features encoding
  - Handling both together
  - Pipeline organization

#### Custom Transformations
- **[Function_Transformer.ipynb](Function_Transformer.ipynb)** - Apply custom functions
  - User-defined transformations
  - FunctionTransformer usage
  - Feature engineering ideas
  - Integration with pipelines

#### Datetime Features
- **[handling_date_time.ipynb](handling_date_time.ipynb)** - Extract datetime information
  - Parse datetime columns
  - Extract day, month, year
  - Create cyclical features
  - Calculate time differences
  - Timezone handling

---

### 🚀 Production-Ready Pipelines

#### Scikit-Learn Pipeline Implementation
- **[productioncode_without_pipeline.ipynb](productioncode_without_pipeline.ipynb)** - Step-by-step preprocessing
  - Prepare data manually
  - Understand each transformation
  - Common mistakes to avoid
  - Visualization of each step

#### Complete Pipeline with Real Data
- **[titanic_with_pipeline.ipynb](titanic_with_pipeline.ipynb)** - Full ML pipeline workflow
  - Load and explore Titanic dataset
  - Apply all preprocessing techniques
  - Build sklearn pipeline
  - Make predictions

- **[titanic_production_code_with_pipeline.ipynb](titanic_production_code_with_pipeline.ipynb)** - Production-grade implementation
  - Handle train-test split properly
  - Prevent data leakage
  - Save and load pipelines
  - Deploy-ready code
  - Error handling

---

### 📊 Model Evaluation

#### Classification Metrics
- **[Classification_metrics.ipynb](Classification_metrics.ipynb)** - Evaluate classifier performance
  - Accuracy, Precision, Recall, F1-Score
  - Confusion matrix
  - ROC-AUC curves
  - Precision-Recall trade-offs
  - When to use which metric

---

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.7+
# Required libraries:
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### Running Notebooks

1. **Clone the repository**
   ```bash
   git clone https://github.com/imharshmishra87/Feature-Scaling-Ml.git
   cd Feature-Scaling-Ml
   ```

2. **Start Jupyter**
   ```bash
   jupyter notebook
   ```

3. **Open any notebook** and run cells to see implementations

### Recommended Learning Path

**Beginner (Understand the Basics):**
1. `handling_missing_values(_simple_imputer).ipynb` - Start simple
2. `feature_scaling(standardization).ipynb` - Learn scaling
3. `Encoding.ipynb` - Handle categorical data
4. `Outlier_detection(IQR).ipynb` - Detect anomalies

**Intermediate (Build Skills):**
1. `knn_imputation.ipynb` - Advanced imputation
2. `powertransformer.ipynb` - Transform distributions
3. `handling_date_time.ipynb` - Work with dates
4. `handling_mixed_var.ipynb` - Handle complex data

**Advanced (Production Code):**
1. `productioncode_without_pipeline.ipynb` - Learn the workflow
2. `titanic_with_pipeline.ipynb` - Build pipeline
3. `titanic_production_code_with_pipeline.ipynb` - Deploy-ready code
4. `Classification_metrics.ipynb` - Evaluate results

---

## 📚 Complete Data Pipeline Flow

```
Raw Data
    ↓
[Exploratory Data Analysis]
    ↓
[Handle Missing Values] ← knn_imputation, simple_imputer, random_imputation, missing_indicator
    ↓
[Detect & Remove Outliers] ← IQR, Z-score, Percentile methods
    ↓
[Encode Categorical Variables] ← Encoding, One-hot, Label encoding
    ↓
[Handle Mixed Variables] ← handling_mixed_var
    ↓
[Feature Scaling] ← Standardization, Normalization, Power transformation
    ↓
[Transform Datetime] ← Date feature extraction
    ↓
[Custom Transformations] ← Function_Transformer
    ↓
[Build ML Pipeline] ← sklearn Pipeline, ColumnTransformer
    ↓
[Train Models & Evaluate] ← Classification metrics
    ↓
Deployment Ready! 🚀
```

---

## 🔧 Techniques Coverage

| Technique | Notebooks | Use Case |
|-----------|-----------|----------|
| **Standardization** | feature_scaling | Features with different scales |
| **Missing Values** | simple_imputer, knn_imputation, random_imputation | Incomplete datasets |
| **Outlier Detection** | IQR, Z-score, Percentile | Identify anomalies |
| **Categorical Encoding** | Encoding | Convert text to numbers |
| **Power Transform** | powertransformer | Normalize skewed data |
| **Datetime Features** | handling_date_time | Extract temporal info |
| **Pipeline Building** | titanic_* | Production workflows |
| **Metrics** | Classification_metrics | Evaluate performance |

---

## 💡 Key Learnings

✅ **Data Quality** - Proper preprocessing improves model performance by 10-30%  
✅ **Avoid Data Leakage** - Fit transformers on train data only  
✅ **Feature Scaling Matters** - Essential for distance-based algorithms  
✅ **Handle Missing Data Wisely** - Don't just delete, impute thoughtfully  
✅ **Pipeline Automation** - Build reusable, maintainable workflows  
✅ **Production Ready** - Know the difference between notebook and production code  

---

## 🎓 Why This Matters

### For ML Success:
- **80% of ML work is data preparation** (Kaggle survey)
- **Proper scaling improves convergence** (Neural networks, SVM, KNN)
- **Pipelines prevent data leakage** (Common interview question)
- **Production code differs from notebooks** (Deployment reality)
- **Understanding preprocessing deepens ML knowledge**

---

## 📊 Real-World Example: Titanic Dataset

Both Titanic notebooks (`titanic_with_pipeline.ipynb` and `titanic_production_code_with_pipeline.ipynb`) demonstrate:

✓ Loading and exploring data  
✓ Identifying missing values  
✓ Handling categorical variables (Sex, Embarked)  
✓ Feature scaling (Fare, Age)  
✓ Building sklearn pipelines  
✓ Training models  
✓ Evaluating with metrics  
✓ Deployment considerations  

**Result:** Complete end-to-end ML workflow!

---

## 🔗 Complete Feature Engineering Checklist

- [ ] Load and explore data
- [ ] Check for missing values
- [ ] Handle missing data (imputation)
- [ ] Identify outliers
- [ ] Remove or cap outliers
- [ ] Encode categorical variables
- [ ] Handle datetime features
- [ ] Scale numerical features
- [ ] Check feature distributions
- [ ] Build pipeline
- [ ] Validate on test set
- [ ] Evaluate metrics
- [ ] Deploy responsibly

---

## 🔄 When to Use Each Technique

### Imputation Methods
- **Mean/Median** - Quick, numerical, not recommended for >20% missing
- **KNN** - Better results, preserves relationships, slower
- **Random** - Avoid bias from single value, add noise
- **Missing Indicator** - Capture the fact that data was missing

### Outlier Detection
- **IQR** - Robust, non-parametric, doesn't assume distribution
- **Z-Score** - For normally distributed data, faster
- **Percentile** - When you want specific cutoffs, interpretable

### Scaling Methods
- **Standardization** - Default choice, better for most algorithms
- **Power Transform** - For skewed distributions, sensitive data
- **No Scaling** - Tree-based models (Random Forest, XGBoost)

---

## 🚀 Technologies Used

- **Python 3** - Programming language
- **Jupyter Notebooks** - Interactive environment
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Scikit-learn** - ML preprocessing and pipelines
- **Matplotlib & Seaborn** - Visualizations

---

## 💬 Common Questions & Answers

**Q: Should I always scale my data?**  
A: No. Tree-based models don't need scaling. Use for linear models, neural networks, SVM, KNN.

**Q: How do I handle extreme outliers?**  
A: Use IQR method (robust), or domain knowledge. Don't blindly delete outliers.

**Q: What's data leakage?**  
A: Using test data information to fit train transformers. Fit on train → transform both train & test.

**Q: Which imputation method is best?**  
A: KNN is good generally, but depends on data. Test multiple approaches.

**Q: How do I avoid overfitting in preprocessing?**  
A: Keep preprocessing simple, use cross-validation, don't fine-tune on test data.

---

## 📈 Expected Improvements

After implementing these techniques properly:
- **10-30%** improvement in model accuracy
- **2-5x faster** training with proper scaling
- **Better generalization** to new data
- **Fewer production errors** with pipelines
- **Reproducible workflows** for team collaboration

---

## 🤝 Contributing

Found a bug or want to add techniques?
- Open an Issue for improvements
- Create a Pull Request for new methods
- Suggest missing preprocessing techniques

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

## 🔗 Related Repositories

🤖 **[Ml-models](https://github.com/imharshmishra87/Ml-models)** - ML algorithm implementations  

---

## 📬 Next Steps

1. **Clone this repo**
2. **Start with beginner notebooks**
3. **Build the Titanic pipeline**
4. **Apply to your own datasets**
5. **Deploy preprocessing pipelines**

---

**Last Updated:** May 2026  
**Status:** Active and maintained  
**Contributions:** Welcome!

⭐ **If this helps you, please star this repository!** 🌟
