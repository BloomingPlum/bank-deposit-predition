# Bank Deposit Prediction - Machine Learning Project

This project aims to predict whether a client will subscribe to a term deposit using various machine learning techniques. The workflow includes exploratory data analysis, preprocessing, model training, hyperparameter optimization, and experiment tracking.

## Project Structure

```
├── 1.EDA.ipynb                       # Exploratory Data Analysis
├── 2.Preprocessing_and modelling.ipynb # Data preprocessing and baseline models
├── 3.XGBoost_and_hyperparameter_tuning.ipynb # XGBoost and hyperparameter tuning
├── 4.Autogluon.ipynb                  # AutoML with AutoGluon
├── 5.Feature_importance_SHAP.ipynb    # Feature importance and SHAP analysis
├── data/                              # Raw and preprocessed datasets
│   ├── bank-additional-full.csv
│   ├── preprocessed_bank_data.csv
│   └── ...
├── models/                            # Saved model artifacts
├── utils/                             # Utility modules
│   ├── eda.py                         # EDA functions
│   ├── eval.py                        # Evaluation functions
│   ├── hyperparam.py                  # Hyperparameter tuning functions
│   ├── mlflow.py                      # MLflow tracking functions
│   └── shap.py                        # SHAP analysis functions
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```

## Requirements

The main dependencies for this project are listed in `requirements.txt`. To install them, run:

```
pip install -r requirements.txt
```

Key packages include:
- pandas
- numpy
- scikit-learn
- xgboost
- optuna
- hyperopt
- autogluon.tabular
- mlflow
- joblib
- seaborn
- matplotlib
- shap

## Workflow Overview

### 1. Exploratory Data Analysis (EDA)
- Load and inspect the dataset.
- Visualize distributions, correlations, and relationships between features.
- Identify missing values and outliers.

### 2. Data Preprocessing & Modeling
- Encode categorical variables and scale numerical features.
- Split data into training and validation sets.
- Train several models and evaluate performance.

### 3. XGBoost & Hyperparameter Tuning
- Train XGBoost models with different hyperparameter optimization frameworks:
  - **Random Search**
  - **Optuna**
  - **Hyperopt**
- Evaluate models using metrics such as F1 score, ROC AUC, precision, and recall.
- Visualize confusion matrices and ROC curves.
- Track experiments and metrics using MLflow.

### 4. AutoML with AutoGluon
- Apply AutoGluon for automated model selection and hyperparameter tuning.
- Compare AutoGluon results with manual approaches.

## How to Run

1. **Install Dependencies**
   - Ensure you have Python 3.8+ and install required packages:
     ```
     pip install -r requirements.txt
     ```
   - Additional packages: `xgboost`, `optuna`, `hyperopt`, `mlflow`, `autogluon`, `joblib`, `seaborn`, `matplotlib`, `scikit-learn`, etc.

2. **Prepare Data**
   - Place the raw dataset (`bank-additional-full.csv`) in the `data/` directory.
   - Run the notebooks in order for preprocessing and model training.

3. **Track Experiments**
   - MLflow is used for experiment tracking. The tracking URI is set to the local `mlruns/` directory.
   - You can launch the MLflow UI with:
     ```
     mlflow ui --backend-store-uri file:./mlruns
     ```
4. **Reproducibility**
   - Preprocessed data and model artifacts are saved in the `data/` and `AutogluonModels/` directories for reproducibility.

## Results
The following table summarizes the F1 scores for all models on both the training and validation sets, as reported in the experiments:

| Model/Method                              | F1 Score (Train) | F1 Score (Validation) |
|:-----------------------------------------:|:----------------:|:---------------------:|
| Poly LR (degree 2)                        |      0.46        |         0.47          |
| Decision Tree                             |      0.47        |         0.48          |
| Random Forest                             |      0.51        |         0.51          |
| XGBoost (RandomSearch)                    |      0.49        |         0.51          |
| XGBoost (Optuna)                          |      0.49        |         0.50          |
| XGBoost (Hyperopt)                        |      0.50        |         0.52          |
| AutoGluon (WeightedEnsemble_L2_FULL)      |      0.53        |         0.55          |


## Future Work
- **Feature Engineering:**  Explore advanced feature engineering techniques to better represent human-driven signals such as contact, poutcome, and month. This may include interaction terms, polynomial features, or domain-specific transformations. Creating composite features like confidence × contact could help the model capture more nuanced relationships. 
- **Class Imbalance Handling:** Investigate more sophisticated approaches for addressing class imbalance, such as SMOTE (Synthetic Minority Oversampling Technique) or ADASYN. 
- **Adaptive Thresholding:** Explore the use of adaptive thresholds based on specific user profiles, segment performance, or cost-sensitive scenarios rather than applying a global fixed value.



