from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('current_diabetes_binary_health_indicators_geometric_BRFSS2023.csv')

# Extract the predictor and target variables
X = data['MetabolicDisorderWithoutBMI']
y = data['AtRisk']

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y, X)

# Calculate Youden's J statistic to find the optimal threshold
youden_j = tpr - fpr
optimal_idx = youden_j.argmax()
optimal_threshold = thresholds[optimal_idx]

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y, X):.2f})')
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f'Optimal Threshold = {optimal_threshold:.2f}')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random chance
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Diabetes Classification')
plt.legend()
plt.show()

print(f"Optimal Threshold for Diabetic State Transition: {optimal_threshold:.2f}")
