"""
SVM Model Training Script
This script trains a Support Vector Machine classifier on the Iris dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
print("Loading dataset...")
from sklearn.datasets import load_iris
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Save dataset for reference
df.to_csv('iris_dataset.csv', index=False)
print(f"Dataset shape: {df.shape}")
print("\nDataset preview:")
print(df.head())

# Prepare features and target
X = df[iris.feature_names]
y = df['species']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*50)
print("Training SVM Models with Different Kernels")
print("="*50)

# Train models with different kernels
kernels = ['linear', 'rbf', 'poly']
models = {}
results = {}

for kernel in kernels:
    print(f"\n--- Training SVM with {kernel.upper()} kernel ---")

    if kernel == 'poly':
        model = SVC(kernel=kernel, degree=3, random_state=42)
    else:
        model = SVC(kernel=kernel, random_state=42)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    models[kernel] = model
    results[kernel] = accuracy

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Select best model
best_kernel = max(results, key=results.get)
best_model = models[best_kernel]

print("\n" + "="*50)
print(f"Best Model: SVM with {best_kernel.upper()} kernel")
print(f"Best Accuracy: {results[best_kernel]:.4f}")
print("="*50)

# Save the best model and scaler
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save feature names
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(iris.feature_names, f)

# Save target names
with open('target_names.pkl', 'wb') as f:
    pickle.dump(iris.target_names, f)

print("\nModel saved as 'svm_model.pkl'")
print("Scaler saved as 'scaler.pkl'")
print("Feature names saved as 'feature_names.pkl'")
print("Target names saved as 'target_names.pkl'")

# Visualize results
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values())
plt.xlabel('Kernel')
plt.ylabel('Accuracy')
plt.title('SVM Performance with Different Kernels')
plt.ylim([0.8, 1.0])
for i, (kernel, acc) in enumerate(results.items()):
    plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
plt.savefig('model_comparison.png')
print("\nComparison plot saved as 'model_comparison.png'")
