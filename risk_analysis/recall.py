import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

data = pd.read_csv('new_diabetes_binary_health_indicators_geometric_BRFSS2023.csv')

# Extract the predictor and target variables
X = data['MetabolicDisorderWithoutBMI']
y = data['Diabetes_binary']

def evaluate_thresholds(y_true, scores, thresholds):
    precisions = []
    recalls = []
    f1_scores = []

    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        precisions.append(precision_score(y_true, predictions))
        recalls.append(recall_score(y_true, predictions))
        f1_scores.append(f1_score(y_true, predictions))

    return precisions, recalls, f1_scores

# Define a range of thresholds
thresholds = np.arange(0.1, 2.1, 0.1)

# Calculate precision, recall, and F1 scores for each threshold
precisions, recalls, f1_scores = evaluate_thresholds(y, X, thresholds)

# Plot Precision-Recall and F1-Score Curves
plt.figure(figsize=(12, 6))

plt.plot(thresholds, precisions, label='Precision', marker='o')
plt.plot(thresholds, recalls, label='Recall', marker='o')
plt.plot(thresholds, f1_scores, label='F1 Score', marker='o')

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision, Recall, and F1 Score at Different Thresholds")
plt.legend()
plt.grid()
plt.show()

# Find the optimal threshold based on your specific goal (e.g., maximizing recall)
optimal_threshold = thresholds[np.argmax(recalls)]
print(f"Optimal Threshold for Maximum Recall: {optimal_threshold}")