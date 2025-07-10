# %pip install lime shap xgboost matplotlib seaborn scikit-learn pandas

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import shap                          # SHAP for model explainability
import lime.lime_tabular as lime    # LIME for local explanations
from xgboost import XGBClassifier    # XGBoost classifier

import warnings
warnings.filterwarnings('ignore')

# Ensure inline plotting in notebooks
%matplotlib inline

# Load breast cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the XGBoost classifier
model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
print("Test Accuracy:", round(accuracy_score(y_test, y_pred), 3))

# SHAP explanation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)

# SHAP summary plots
shap.summary_plot(shap_values, features=X_test, feature_names=data.feature_names)
shap.summary_plot(shap_values, features=X_test, feature_names=data.feature_names, plot_type='beeswarm')

# LIME explanation for a single instance
explainer_lime = lime.LimeTabularExplainer(
    X_train_scaled,
    feature_names=data.feature_names,
    class_names=data.target_names,
    discretize_continuous=True
)

idx = 5  # Sample index
exp = explainer_lime.explain_instance(
    X_test_scaled[idx],
    model.predict_proba,
    num_features=10
)
exp.show_in_notebook(show_table=True)

# Examine misclassified instances
wrong_idx = np.where(y_pred != y_test)[0]
print(f"Number of misclassified samples: {len(wrong_idx)}")

if len(wrong_idx):
    i = wrong_idx[0]
    print(f"True label: {data.target_names[y_test.iloc[i]]}, Prediction: {data.target_names[y_pred[i]]}")
    exp_wrong = explainer_lime.explain_instance(
        X_test_scaled[i],
        model.predict_proba,
        num_features=10
    )
    exp_wrong.show_in_notebook(show_table=True)
else:
    print("The model correctly classified all test samples!")

# SHAP dependence plot for a specific feature
shap.dependence_plot(
    "mean radius",
    shap_values,
    X_test,
    feature_names=data.feature_names
)