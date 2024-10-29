import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the data
data = pd.read_csv('new_diabetes_binary_health_indicators_geometric_BRFSS2023.csv')

# Step 2: Set target variable
y = data['Diabetes_binary']

# Step 3: List of independent variables for univariable analysis
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
    'MetabolicDisorderWithoutBMI',
    'NewMetabolicDisorder'
]

# Step 4: Prepare a DataFrame to store the results
results_df = pd.DataFrame(columns=['Variable', 'Coefficient', 'Odds_Ratio', 'P_Value', 'CI_Lower', 'CI_Upper'])

# Step 5: Perform univariable logistic regression for each independent variable
for variable in independent_variables:
    # Step 5a: Define the model
    X = sm.add_constant(data[[variable]])  # Add constant for intercept
    
    # Step 5b: Fit the model
    model = sm.Logit(y, X)
    result = model.fit(disp=0)  # disp=0 to avoid printing fitting messages
    
    # Step 5c: Extract key metrics
    coef = result.params[1]  # Get the coefficient of the variable
    odds_ratio = np.exp(coef)  # Calculate odds ratio
    p_value = result.pvalues[1]  # Get the p-value
    conf_int = result.conf_int().iloc[1]  # Get the confidence interval for the variable
    
    # Step 5d: Append results to DataFrame
    results_df = results_df._append({
        'Variable': variable,
        'Coefficient': coef,
        'Odds_Ratio': odds_ratio,
        'P_Value': p_value,
        'CI_Lower': conf_int[0],
        'CI_Upper': conf_int[1],
    }, ignore_index=True)

# Step 6: Display the results DataFrame
print(results_df)

# Step 7: Plot Odds Ratios
plt.figure(figsize=(13, 6))
sns.barplot(data=results_df, x='Odds_Ratio', y='Variable', ci=None, palette='viridis')
plt.axvline(x=1, linestyle='--', color='red')  # Reference line at OR = 1
plt.title('Odds Ratios for Diabetes Risk Factors')
plt.xlabel('Odds Ratio')
plt.ylabel('Variables')

plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.1)

plt.show()
