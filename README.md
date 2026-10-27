# Customer Churn Prediction & Customer Segmentation

## Project Overview

This project focuses on analyzing customer behavior, identifying patterns associated with customer churn, and developing machine learning models to predict whether a customer is likely to leave a service.

Using a customer churn dataset, the project combines **Exploratory Data Analysis (EDA), data preprocessing, data visualization, supervised machine learning, dimensionality reduction, and unsupervised clustering** to explore customer behavior and build predictive models.

The primary objective is to support data-driven customer retention strategies by identifying customers who may be at risk of leaving.

## Objectives

- Analyze customer demographics, account balances, and banking behavior.
- Explore relationships between customer attributes and churn.
- Clean and preprocess data for machine learning.
- Build and evaluate classification models for churn prediction.
- Tune a Random Forest classifier using `GridSearchCV`.
- Apply PCA for dimensionality reduction and K-Means for customer segmentation.
- Explore regression models and a neural network as additional machine learning approaches.
- Save trained models for future inference.

## Dataset

The project uses `Churn.csv`, a customer churn dataset containing demographic, financial, and behavioral information.

Important features include:

| Feature | Description |
|---|---|
| `CreditScore` | Customer's credit score |
| `Geography` | Customer's geographic region |
| `Gender` | Customer's gender |
| `Age` | Customer's age |
| `Tenure` | Number of years the customer has been with the bank |
| `Balance` | Customer's account balance |
| `NumOfProducts` | Number of products held by the customer |
| `HasCrCard` | Whether the customer has a credit card |
| `IsActiveMember` | Whether the customer is an active member |
| `EstimatedSalary` | Estimated customer salary |
| `Exited` | Target variable indicating customer churn |

The target variable is `Exited`:

- `0` — Customer retained
- `1` — Customer exited

The columns `RowNumber`, `CustomerId`, and `Surname` are excluded from the modeling dataset because they are identifiers or non-predictive text fields.

## Project Workflow

### 1. Exploratory Data Analysis

The notebook investigates customer characteristics and churn patterns using statistical summaries and visualizations, including:

- Target class distribution
- Customer age and credit score distributions
- Account balance by geography and churn status
- Correlation heatmap
- Pairwise relationships between numerical features
- Churn rates across product counts and gender
- Salary and balance relationships across customer groups

### 2. Data Preprocessing

The preprocessing workflow includes:

- Handling missing values using forward-fill and backward-fill.
- Encoding categorical variables, including gender and geography.
- Removing identifier columns and redundant features.
- Separating the feature matrix from the target variable.
- Splitting the dataset into training and validation sets using stratified sampling.
- Standardizing features for models that benefit from scaling.

### 3. Dimensionality Reduction

Principal Component Analysis (PCA) is applied to the standardized training features to reduce the data to two dimensions and examine its underlying structure.

### 4. Machine Learning Models

The notebook explores several modeling approaches.

**Classification**
- Logistic Regression
- Support Vector Classifier (SVC)
- Random Forest Classifier
- XGBoost, when available
- Artificial Neural Network (ANN), when TensorFlow is available

**Regression experiments**
- Linear Regression
- Support Vector Regression (SVR)
- Decision Tree Regressor
- Random Forest Regressor

**Customer segmentation**
- K-Means clustering with three clusters

Classification is the primary approach for predicting customer churn, while regression and clustering are included as additional experiments.

### 5. Hyperparameter Tuning

`GridSearchCV` is used to tune the Random Forest classifier with three-fold cross-validation.

The parameter grid explores:

- `n_estimators`: 50 and 100
- `max_depth`: 5 and 10
- `min_samples_split`: 2 and 5

The search uses **ROC-AUC** as its optimization metric to select the best estimator.

### 6. Model Evaluation

The notebook evaluates classification performance using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Five-fold cross-validation for Logistic Regression
- ROC-AUC for Random Forest hyperparameter selection
- Comparison of model accuracy and error rates

These metrics provide complementary views of model performance, particularly when identifying customers at risk of churn.

### 7. Model Persistence

The final trained artifacts are saved using `joblib`:

| File | Purpose |
|---|---|
| `churn_final_model.joblib` | Tuned Random Forest classifier |
| `feature_scaler.joblib` | StandardScaler used during preprocessing |
| `churn_kmeans_cluster.joblib` | K-Means clustering model |

The notebook also loads the saved Random Forest model and performs a sample prediction with a churn probability.

## Technologies Used

- **Python** — Core programming language
- **Pandas & NumPy** — Data manipulation and numerical computing
- **Matplotlib & Seaborn** — Data visualization
- **Scikit-learn** — Preprocessing, machine learning, model evaluation, PCA, K-Means, and hyperparameter tuning
- **XGBoost** — Gradient boosting classification, when available
- **TensorFlow / Keras** — Neural network experiments, when available
- **Joblib** — Saving and loading trained models
- **Jupyter Notebook** — Interactive analysis and experimentation

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction-ml.git
cd customer-churn-prediction-ml
```

Install the core dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
```

For the optional XGBoost and neural network experiments:

```bash
pip install xgboost tensorflow
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebook and ensure that `Churn.csv` is available in the working directory before running the data-loading cells.

## Repository Structure

```text
customer-churn-prediction-ml/
│
├── code.ipynb
├── Churn.csv
├── churn_final_model.joblib
├── feature_scaler.joblib
├── churn_kmeans_cluster.joblib
└── README.md
```

The dataset and model artifacts may be excluded from the repository if they are not intended for distribution. The model files are generated when the relevant notebook cells run successfully.

## Potential Applications

- Identifying customers with a higher likelihood of leaving
- Supporting customer retention campaigns
- Understanding behavioral patterns associated with churn
- Segmenting customers for targeted marketing strategies
- Supporting data-driven customer relationship management

## Future Improvements

- Use a scikit-learn `Pipeline` to combine preprocessing and model training.
- Fit imputers and encoders only on training data to reduce data leakage.
- Compare models using ROC-AUC, precision-recall AUC, and recall at a chosen threshold.
- Address class imbalance using class weights or resampling when appropriate.
- Tune the decision threshold based on the business cost of customer churn.
- Apply feature importance and SHAP analysis to improve model interpretability.
- Evaluate clustering quality and interpret the resulting customer segments.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for customer churn analysis, from exploratory data analysis and preprocessing to classification, model evaluation, clustering, and model persistence.

By combining predictive modeling with customer behavior analysis, the project provides a foundation for understanding churn patterns and developing more informed customer retention strategies.
