import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_csv('diabetes_binary_health_indicators_geometric_BRFSS2023.csv')

# Separate the data into diabetic and non-diabetic cases
diabetic = data[data['Diabetes_binary'] == 1]['MetabolicDisorderWithoutBMI']
non_diabetic = data[data['Diabetes_binary'] == 0]['MetabolicDisorderWithoutBMI']

# Set up the plot
plt.figure(figsize=(12, 6))

# Plot density plots for both groups
sns.kdeplot(diabetic, shade=True, color='r', label='Diabetic')
sns.kdeplot(non_diabetic, shade=True, color='b', label='Non-Diabetic')

# Add a vertical line at the identified threshold
plt.axvline(x=1.41, color='green', linestyle='--', label='Optimal Threshold (1.41)')

# Labeling the plot
plt.title('Density Plot of Metabolic Disorder without BMI for Diabetic vs Non-Diabetic Cases')
plt.xlabel('Metabolic Disorder without BMI')
plt.ylabel('Density')
plt.legend()

plt.show()
