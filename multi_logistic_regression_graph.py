import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the data
data = pd.read_csv('diabetes_binary_health_indicators_geometric_BRFSS2023.csv')

# Step 2: Set target variable
y = data['Diabetes_binary']

# Step 3: List of independent variables for multivariable analysis
independent_variables = [
    'BMI', 
    'HighBP', 
    'HighChol', 
    'Stroke', 
    'HeartDiseaseorAttack', 
    'PhysActivity', 
    'AnyHealthcare', 
    'Sex', 
    'AgeGroup', 
    'Race', 
    'Education', 
    'Income', 
    'MetabolicDisorderWithBMI',
    'MetabolicDisorderWithoutBMI'
]

# Step 4: Define the feature matrix (X) and add a constant
X = sm.add_constant(data[independent_variables])

# Step 5: Fit the multivariable logistic regression model
model = sm.Logit(y, X)
result = model.fit()

# Step 6: Extract key metrics for each variable
coef = result.params  # Coefficients
odds_ratios = np.exp(coef)  # Odds Ratios
p_values = result.pvalues  # P-values
conf_int = result.conf_int()  # Confidence intervals
conf_int_exp = np.exp(conf_int)  # Exponentiate CI

# Step 7: Prepare the results DataFrame
results_df = pd.DataFrame({
    'Variable': coef.index,
    'Coefficient': coef.values,
    'Odds_Ratio': odds_ratios,
    'P_Value': p_values,
    'CI_Lower': conf_int_exp[0].values,
    'CI_Upper': conf_int_exp[1].values
})

# Step 8: Display the results DataFrame
print(results_df)

# Step 9: Plot Odds Ratios
plt.figure(figsize=(13, 6))
sns.barplot(data=results_df[1:], x='Odds_Ratio', y='Variable', ci=None, palette='viridis')  # Skip constant in plot
plt.axvline(x=1, linestyle='--', color='red')  # Reference line at OR = 1
plt.title('Adjusted Odds Ratios for Diabetes Risk Factors')
plt.xlabel('Odds Ratio')
plt.ylabel('Variables')

plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.1)

plt.show()
