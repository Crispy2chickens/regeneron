# Step 1: Import necessary libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

# Step 2: Load your dataset (assuming it's a CSV file)
brfss_binary = pd.read_csv('diabetes_binary_health_indicators_with_index_BRFSS2023.csv')

# Step 3: Select relevant columns for the composite score
X = brfss_binary[['BMI', 'HighBP', 'HighChol', 'HeartDiseaseorAttack']]
y = brfss_binary['Diabetes_binary']  # Ensure this column is available

# Step 4: Fit logistic regression to determine weights
model = LogisticRegression()
model.fit(X, y)

# Step 5: Extract the coefficients (weights)
weights = model.coef_[0]

# Step 6: Normalize the weights to sum to 1
normalized_weights = weights / np.sum(np.abs(weights))

# Display the weights for each feature
print(f'BMI: {normalized_weights[0]:.2f}, HighBP: {normalized_weights[1]:.2f}, '
      f'HighChol: {normalized_weights[2]:.2f}, HeartDisease: {normalized_weights[3]:.2f}')

# # Step 7: Apply the weights to calculate the composite risk score
# brfss_binary['Risk_Score'] = (
#     brfss_binary['BMI'] * normalized_weights[0] +
#     brfss_binary['HighBP'] * normalized_weights[1] +
#     brfss_binary['HighChol'] * normalized_weights[2] +
#     brfss_binary['HeartDiseaseorAttack'] * normalized_weights[3]
# )

# # Step 8: Save the new DataFrame with the Risk_Score column
# brfss_binary.to_csv('brfss_binary_with_risk_score.csv', index=False)

# # Step 9: Check the distribution of the new Risk_Score column
# print(brfss_binary['Risk_Score'].describe())
