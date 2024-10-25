import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the data
data = pd.read_csv('diabetes_binary_final.csv')

# Step 2: Select features and target
X = data[['BMI', 'Weight', 'HighBP', 'HighChol', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'HvyAlcoholConsump', 
          'PhysActivity', 'GenHlth', 'PhysHlth', 'MentHlth', 'PoorHlth', 'DiffWalk', 
          'AnyHealthcare', 'NoDocbcCost', 'Sex', 'AgeGroup', 'Race', 'Education', 'Income',
          'MetabolicDisorderWithBMI', 'MetabolicDisorderWithoutBMI']]

# Step 3: Add constant to the feature set
X = sm.add_constant(X)  

# Step 4: Set target variable
y = data['Diabetes_binary']

# Step 5: Build and fit the logistic regression model
model = sm.Logit(y, X)
result = model.fit()

# Step 6: Print the model summary
print(result.summary())

# Step 7: Calculate odds ratios and confidence intervals
odds_ratios = pd.DataFrame({
    'OR': np.exp(result.params),  
    'CI_lower': np.exp(result.conf_int()[0]), 
    'CI_upper': np.exp(result.conf_int()[1])   
})

print("Odds Ratios and Confidence Intervals:")
print(odds_ratios)

# Step 8: Define average and high BMI scenarios for prediction
average_bmi = data['BMI'].mean()

avg_bmi_df = pd.DataFrame({
    'const': 1,
    'BMI': [average_bmi],  # Using average BMI from your data
    'Weight': [0],  
    'HighBP': [0], 
    'HighChol': [0], 
    'Smoker': [0], 
    'Stroke': [0], 
    'HeartDiseaseorAttack': [0], 
    'HvyAlcoholConsump': [0], 
    'PhysActivity': [0], 
    'GenHlth': [3],  
    'PhysHlth': [0],  
    'MentHlth': [0],  
    'PoorHlth': [0],  
    'DiffWalk': [0],  
    'AnyHealthcare': [1],  
    'NoDocbcCost': [0],  
    'Sex': [1],  
    'AgeGroup': [8],  
    'Race': [1],  
    'Education': [4],  
    'Income': [3],  
    'MetabolicDisorderWithBMI': [0],
    'MetabolicDisorderWithoutBMI': [0]
})

high_bmi_df = pd.DataFrame({
    'const': 1,
    'BMI': [average_bmi + 5],  # High BMI (5 units above average)
    'Weight': [0],  
    'HighBP': [0], 
    'HighChol': [0], 
    'Smoker': [0], 
    'Stroke': [0], 
    'HeartDiseaseorAttack': [0], 
    'HvyAlcoholConsump': [0], 
    'PhysActivity': [0], 
    'GenHlth': [3],  
    'PhysHlth': [0],  
    'MentHlth': [0],  
    'PoorHlth': [0],  
    'DiffWalk': [0],  
    'AnyHealthcare': [1],  
    'NoDocbcCost': [0],  
    'Sex': [1],  
    'AgeGroup': [8],  
    'Race': [1],  
    'Education': [4],  
    'Income': [3],  
    'MetabolicDisorderWithBMI': [0],
    'MetabolicDisorderWithoutBMI': [0]
})

# Step 9: Ensure columns match the training data (X) for prediction
avg_bmi_df = avg_bmi_df[X.columns]
high_bmi_df = high_bmi_df[X.columns]

# Step 10: Make predictions for average and high BMI
predicted_probability_avg_bmi = result.predict(avg_bmi_df)
predicted_probability_high_bmi = result.predict(high_bmi_df)

# Step 11: Calculate Attributable Risk (AR) for BMI
AR = predicted_probability_high_bmi - predicted_probability_avg_bmi
print(f"Attributable Risk for BMI: {AR[0]:.4f}")

# Step 12: Calculate the proportion of risk attributable to BMI
total_odds_ratio = odds_ratios['OR'].sum()
proportion_attributable_to_bmi = odds_ratios.loc['BMI', 'OR'] / total_odds_ratio
print(f"Proportion of Risk Attributable to BMI: {proportion_attributable_to_bmi:.4f}")

# Step 13: Plot Odds Ratios
plt.figure(figsize=(9, 8))
sns.barplot(x=odds_ratios.index, y='OR', data=odds_ratios)
plt.xticks(rotation=90, fontsize=8)
plt.subplots_adjust(top=0.9, bottom=0.23)
plt.title('Odds Ratios for Diabetes Risk Factors')
plt.ylabel('Odds Ratio')
plt.xlabel('Predictors')
plt.axhline(1, linestyle='--', color='red')  # Line at OR=1 for reference
plt.show()

# Step 14: Summarize significant predictors (OR > 1)
significant_predictors = odds_ratios[odds_ratios['OR'] > 1].sort_values(by='OR', ascending=False)
print("Significant Predictors:")
print(significant_predictors)
