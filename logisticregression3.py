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
X = data[['BMI', 'Weight', 'HighBP', 'HighChol', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'HvyAlcoholConsump', 
          'PhysActivity', 'GenHlth', 'PhysHlth', 'MentHlth', 'PoorHlth', 'DiffWalk', 
          'AnyHealthcare', 'NoDocbcCost', 'Sex', 'AgeGroup', 'Race', 'Education', 'Income',
          'MetabolicDisorderWithBMI', 'MetabolicDisorderWithoutBMI']]

y = data['Diabetes_binary']

# Step 3: Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Step 4: Fit the logistic regression model
X = sm.add_constant(X)  # Adds a constant term to the predictor
model = sm.Logit(y, X)
result = model.fit()

# Step 5: Calculate odds ratios
odds_ratios = np.exp(result.params)

# Step 6: Plot odds ratios
plt.figure(figsize=(10, 6))
sns.barplot(x=odds_ratios.index, y=odds_ratios.values)
plt.xticks(rotation=90)
plt.title('Odds Ratios for Diabetes Risk Factors')
plt.xlabel('Features')
plt.ylabel('Odds Ratio')
plt.axhline(1, color='red', linestyle='--')  # Reference line at 1
plt.tight_layout()
plt.show()
