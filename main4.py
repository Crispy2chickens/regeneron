from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the data
data = pd.read_csv('diabetes_binary_final.csv')

# Step 2: Prepare the data
data['Diabetes_binary'] = data['Diabetes_binary'].astype(int)

# Select features and target
X = data[['BMI', 'Smoker', 'HvyAlcoholConsump', 'PhysActivity', 
           'GenHlth', 'PhysHlth', 'MentHlth', 'PoorHlth', 'DiffWalk', 
           'AnyHealthcare', 'NoDocbcCost', 'Sex', 'AgeGroup', 
           'Race', 'Education', 'Income', 'MetabolicDisorderWithBMI']]

y = data['Diabetes_binary']

# Step 3: Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Step 4: Calculate VIF
X = sm.add_constant(X)  # Adds a constant term to the predictors
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Display VIF values
print(vif_data)

# Step 5: Fit the logistic regression model
model = sm.Logit(y, X)
result = model.fit()

# Step 6: Calculate odds ratios
odds_ratios = np.exp(result.params)

# Step 7: Plot odds ratios
plt.figure(figsize=(10, 6))
sns.barplot(x=odds_ratios.index, y=odds_ratios.values)
plt.xticks(rotation=90)
plt.title('Odds Ratios for Diabetes Risk Factors')
plt.xlabel('Features')
plt.ylabel('Odds Ratio')
plt.axhline(1, color='red', linestyle='--')  # Reference line at 1
plt.tight_layout()
plt.show()
