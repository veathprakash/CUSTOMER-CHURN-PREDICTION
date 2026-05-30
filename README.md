# CUSTOMER-CHURN-PREDICTION
This project builds an end-to-end machine learning pipeline to identify customers at risk of churning — before they leave. By combining predictive modeling with explainable AI, it enables business teams to take data-driven, proactive retention action.

Overview

This project implements an end-to-end machine learning pipeline to predict customer churn using a Random Forest Classifier. It helps businesses identify at-risk customers in advance, enabling targeted retention strategies that reduce revenue loss and improve customer lifetime value.

Features

Exploratory Data Analysis (EDA) with visual insights
Data preprocessing — handling missing values, encoding, scaling
Feature engineering — recency, frequency & engagement patterns
Model training using Random Forest Classifier
Hyperparameter tuning with GridSearchCV / RandomizedSearchCV
Feature importance visualization using built-in RF importance & SHAP
Evaluation — AUC-ROC, Precision, Recall, F1-Score, Confusion Matrix
Model

The core model is a RandomForestClassifier from scikit-learn. Random Forest was chosen for its robustness to overfitting, ability to handle mixed data types, and built-in feature importance — making it highly interpretable for business stakeholders.

Tech stack

pandas · numpy — data processing
scikit-learn — Random Forest model & evaluation
shap — explainability & feature importance
matplotlib · seaborn — visualization
joblib — model serialization

customer-churn-prediction/
├── data/
│   ├── raw/               # Original dataset
│   └── processed/         # Cleaned & engineered data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── models/                # Saved model files
├── requirements.txt
└── README.md

Results

The Random Forest Classifier delivered strong predictive performance. Feature importance analysis highlighted engagement recency, support ticket history, and subscription duration as the top drivers of churn. SHAP values further validated these findings with per-customer level explanations.

License

This project is licensed under the MIT License. Feel free to use, modify, and distribute.
