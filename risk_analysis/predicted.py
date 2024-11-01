import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

# Load the dataset
data = pd.read_csv('new_diabetes_binary_health_indicators_geometric_BRFSS2023.csv')

# Define your features (X) and target variable (y)
X = data.drop(columns=['Diabetes_binary'])  # Adjust as per your feature set
y = data['Diabetes_binary']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model (e.g., Logistic Regression)
model = LogisticRegression()  # You can replace this with your model
model.fit(X_train, y_train)

# Get predicted probabilities
y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

# Visualize the distribution of predicted probabilities
plt.figure(figsize=(10, 6))
sns.histplot(y_prob, bins=30, kde=True)
plt.title('Distribution of Predicted Probabilities for Diabetes')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.axvline(x=0.5, color='red', linestyle='--', label='Default Threshold (0.5)')
plt.legend()
plt.show()
