import pandas as pd
import numpy as np
import statsmodels.api as sm

data = pd.read_csv('new_diabetes_binary_health_indicators_geometric_BRFSS2023.csv')

y = data['Diabetes_binary']

# Step 1: Define the model for MetabolicDisorderWithoutBMI
X_metabolic_without_bmi = sm.add_constant(data[['MetabolicDisorderWithoutBMI']])
model_metabolic_without_bmi = sm.Logit(y, X_metabolic_without_bmi)
result_metabolic_without_bmi = model_metabolic_without_bmi.fit(disp=0)

# Step 2: Generate a range of values for MetabolicDisorderWithoutBMI
metabolic_values = np.linspace(data['MetabolicDisorderWithoutBMI'].min(), data['MetabolicDisorderWithoutBMI'].max(), 100)

# Step 3: Calculate predicted probabilities for each value in this range
X_pred = sm.add_constant(pd.DataFrame({'MetabolicDisorderWithoutBMI': metabolic_values}))
predicted_probs = result_metabolic_without_bmi.predict(X_pred)

# Step 4: Find the threshold where the predicted probability crosses 0.5
threshold_metabolic_without_bmi = metabolic_values[np.where(predicted_probs >= 0.5)[0][0]]

print(f"The threshold for MetabolicDisorderWithoutBMI at which diabetes probability reaches 0.5 is approximately: {threshold_metabolic_without_bmi}")
