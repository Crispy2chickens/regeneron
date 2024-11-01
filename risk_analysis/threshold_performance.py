from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd

data = pd.read_csv('new_diabetes_binary_health_indicators_geometric_BRFSS2023.csv')

# Extract the predictor and target variables
X = data['MetabolicDisorderWithoutBMI']
y = data['Diabetes_binary']

threshold = 1.41

# Define a function to evaluate performance at a specific threshold
def evaluate_threshold(y_true, scores, threshold):
    # Binarize scores based on the threshold
    predictions = (scores >= threshold).astype(int)
    
    # Calculate metrics
    conf_matrix = confusion_matrix(y_true, predictions)
    precision = precision_score(y_true, predictions)
    recall = recall_score(y_true, predictions)
    f1 = f1_score(y_true, predictions)
    
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall (Sensitivity): {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
# Apply it to your data
evaluate_threshold(y, X, threshold)
