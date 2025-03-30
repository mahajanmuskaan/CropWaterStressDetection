import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier  # XGBoost Model

# Load Dataset
df = pd.read_csv("../data/Semi_Crop_Water_Stress_Dataset.csv")

# Remove duplicates and fill missing values
df.drop_duplicates(inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

# Encode Target Variable
le = LabelEncoder()
df["Water Stress Level"] = le.fit_transform(df["Water Stress Level"])

# Feature Selection
X = df[['Temperature (°C)', 'Humidity (%)', 'Soil Moisture (%)', 'Rainfall (mm)', 'NDVI', 'Wind Speed (m/s)']]
y = df["Water Stress Level"]

# Scale Features - **Do this before SMOTE**
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance dataset - **Apply it after scaling**
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check class distribution in training and test sets
print("Class distribution in y_train:", y_train_resampled.value_counts())
print("Class distribution in y_test:", y_test.value_counts())

# Define Models
models = {
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42, class_weight='balanced'),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    'XGBoost': XGBClassifier(random_state=42)
}

# Hyperparameter Tuning using RandomizedSearchCV for each model
param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance']
    },
    'Decision Tree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7]
    }
}

# Hyperparameter tuning using RandomizedSearchCV for each model
best_models = {}
for model_name, model in models.items():
    print(f"Hyperparameter tuning for {model_name}...")
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grids[model_name], n_iter=50, cv=3, verbose=2, random_state=42)
    random_search.fit(X_train_resampled, y_train_resampled)
    best_models[model_name] = random_search.best_estimator_

# Evaluate all models
best_accuracy = 0
best_model = None
for model_name, model in best_models.items():
    # Use scaled data for evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Best Model evaluation
print(f"\nBest Model: {best_model}")
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Model Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot Confusion Matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the Best Model and Scaler
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(cm, "confusion_matrix.pkl")

# Display SHAP Summary Plot (Optional)
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Generate SHAP Summary Plot for each class in multi-class classification
for i in range(len(shap_values)):
    print(f"SHAP Summary Plot for Class {i}")
    shap.summary_plot(shap_values[i], X_test, feature_names=X.columns)
