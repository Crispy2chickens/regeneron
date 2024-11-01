from sklearn.metrics import confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('new_diabetes_binary_health_indicators_geometric_BRFSS2023.csv')

# Define a range of alternative thresholds
thresholds = np.arange(0.3, 0.9, 0.1)

# Initialize a list to store sensitivity, specificity, and AUC for each threshold
results = []

# Assuming you have a column 'PredictedProbabilities' containing the predicted probabilities
# Replace 'PredictedProbabilities' with the actual column name if it's different
predicted_probabilities = data['MetabolicDisorderWithoutBMI']

for threshold in thresholds:
    # Classify as diabetic if 'Metabolic Disorder without BMI' >= threshold
    predictions = (predicted_probabilities >= threshold).astype(int)
    
    # Calculate confusion matrix values
    tn, fp, fn, tp = confusion_matrix(data['Diabetes_binary'], predictions).ravel()
    
    # Calculate sensitivity and specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True positive rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate
    
    # Calculate AUC using predicted probabilities
    auc = roc_auc_score(data['Diabetes_binary'], predicted_probabilities)

    # Store results
    results.append({
        'Threshold': threshold,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'AUC': auc
    })

# Convert results to a DataFrame for easy viewing
results_df = pd.DataFrame(results)

# Display the results
print(results_df)
