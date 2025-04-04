import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import optuna
import itertools
import pickle
from tabulate import tabulate

# 1. Load the data
train_path = "/Users/dayanandks/Desktop/final_proj_1/backend/archive/Train_data.csv"
test_path = "/Users/dayanandks/Desktop/final_proj_1/backend/archive/Test_data.csv"

if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise FileNotFoundError("Train or Test dataset not found!")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# 2. Exploratory Data Analysis
print("Data Information:")
print(train.info())
print("\nData Description:")
print(train.describe())

# Check for missing values
print("\nMissing Values:")
print(train.isnull().sum())

# Check for duplicates
print(f"\nNumber of duplicate rows: {train.duplicated().sum()}")

# Visualize class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x=train['class'])
plt.title('Class Distribution')
plt.savefig('class_distribution.png')
plt.show()

# 3. Data Preprocessing
# Encode categorical features
def encode_categorical(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])
    return df

train = encode_categorical(train)
test = encode_categorical(test)

# Remove unnecessary columns
if 'num_outbound_cmds' in train.columns:
    train.drop(['num_outbound_cmds'], axis=1, inplace=True)
    test.drop(['num_outbound_cmds'], axis=1, inplace=True)

# Feature selection
X_train = train.drop(['class'], axis=1)
Y_train = train['class']

# Using Random Forest Classifier for feature selection
rfc = RandomForestClassifier(random_state=42)
rfe = RFE(rfc, n_features_to_select=10)
rfe = rfe.fit(X_train, Y_train)

# Get selected features
selected_features = [v for i, v in zip(rfe.get_support(), X_train.columns) if i]
print("\nSelected Features:")
print(selected_features)

# Use only selected features
X_train = X_train[selected_features]

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.transform(test[selected_features])

# Split the data
x_train, x_test, y_train, y_test = train_test_split(X_train_scaled, Y_train, train_size=0.70, random_state=42)

# 4. Model Training and Hyperparameter Tuning

# KNN Hyperparameter Tuning
def objective_knn(trial):
    n_neighbors = trial.suggest_int('KNN_n_neighbors', 2, 16)
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(x_train, y_train)
    return classifier.score(x_test, y_test)

study_knn = optuna.create_study(direction='maximize')
study_knn.optimize(objective_knn, n_trials=30)
print("\nBest KNN Parameters:", study_knn.best_trial.params)

# Decision Tree Hyperparameter Tuning
def objective_dt(trial):
    max_depth = trial.suggest_int('dt_max_depth', 2, 32)
    max_features = trial.suggest_int('dt_max_features', 2, 10)
    classifier = DecisionTreeClassifier(max_features=max_features, max_depth=max_depth, random_state=42)
    classifier.fit(x_train, y_train)
    return classifier.score(x_test, y_test)

study_dt = optuna.create_study(direction='maximize')
study_dt.optimize(objective_dt, n_trials=30)
print("Best Decision Tree Parameters:", study_dt.best_trial.params)

# 5. Model Training with Best Parameters
models = {
    'KNN': KNeighborsClassifier(n_neighbors=study_knn.best_trial.params['KNN_n_neighbors']),
    'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(
        max_features=study_dt.best_trial.params['dt_max_features'], 
        max_depth=study_dt.best_trial.params['dt_max_depth'],
        random_state=42
    )
}

# Train models and store scores
train_scores = {}
test_scores = {}
predictions = {}

for name, model in models.items():
    model.fit(x_train, y_train)
    train_scores[name] = model.score(x_train, y_train)
    test_scores[name] = model.score(x_test, y_test)
    predictions[name] = model.predict(x_test)

# 6. Model Evaluation
print("\nModel Performance:")
model_performance = [
    [name, train_scores[name], test_scores[name]] 
    for name in models.keys()
]
print(tabulate(model_performance, headers=["Model", "Train Score", "Test Score"], tablefmt="fancy_grid"))

# Cross-validation scores
print("\nCross-Validation Results:")
for name, model in models.items():
    precision = cross_val_score(model, x_train, y_train, cv=10, scoring='precision').mean() * 100
    recall = cross_val_score(model, x_train, y_train, cv=10, scoring='recall').mean() * 100
    print(f"{name}:")
    print(f"  Precision: {precision:.2f}%")
    print(f"  Recall: {recall:.2f}%")

# Detailed classification reports
target_names = ["normal", "anomaly"]  # Adjust based on your dataset
for name, pred in predictions.items():
    print(f"\n{name} Classification Report:")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=target_names))

# F1 Scores
f1_scores = {name: f1_score(y_test, pred) for name, pred in predictions.items()}
plt.figure(figsize=(10, 6))
plt.bar(f1_scores.keys(), f1_scores.values())
plt.title('F1 Scores by Model')
plt.ylim(0.8, 1.0)
plt.savefig('f1_scores.png')
plt.show()

# 7. Save the best model
best_model_name = max(f1_scores, key=f1_scores.get)
best_model = models[best_model_name]

# Ensure the model directory exists
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Save the best model
model_path = os.path.join(model_dir, "best_intrusion_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)
print(f"Best model '{best_model_name}' saved successfully at: {model_path}")

# 8. Load & Test the Model (Optional)
def load_and_test_model(test_data):
    try:
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
        predictions = loaded_model.predict(test_data)
        return predictions
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Example usage:
# new_predictions = load_and_test_model(test_scaled)
