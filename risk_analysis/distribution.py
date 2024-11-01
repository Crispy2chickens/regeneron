import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_csv('new_diabetes_binary_health_indicators_geometric_BRFSS2023.csv')

# Separate the data into diabetic and non-diabetic cases
old = data['MetabolicDisorderWithoutBMI']
new = data['NewMetabolicDisorder']

# Set up the plot
plt.figure(figsize=(12, 6))

# Plot density plots for both groups
sns.kdeplot(old, shade=True, color='r', label='Old')
sns.kdeplot(new, shade=True, color='b', label='New')

# Labeling the plot
plt.title('Density Plot of Old Metric vs New Metric')
plt.xlabel('Metabolic Disorder without BMI')
plt.ylabel('Density')
plt.legend()

plt.show()
