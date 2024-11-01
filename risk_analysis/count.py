import pandas as pd

# Load the dataset
data = pd.read_csv('new_diabetes_binary_health_indicators_geometric_BRFSS2023.csv')

print(data.groupby(['Diabetes_binary']).size())

# threshold = 1.41
threshold = 1.19

# Count individuals based on 'Metabolic Disorder without BMI' threshold
metabolic_disorder_less_1 = data[data['MetabolicDisorderWithoutBMI'] < threshold].shape[0]
metabolic_disorder_gte_1 = data[data['MetabolicDisorderWithoutBMI'] >= threshold].shape[0]

new_metabolic_disorder_less_1 = data[data['NewMetabolicDisorder'] < threshold].shape[0]
new_metabolic_disorder_gte_1 = data[data['NewMetabolicDisorder'] >= threshold].shape[0]

# Count individuals with 'Diabetes_binary' = 0 and 'Diabetes_binary' = 1 within each threshold category
diabetes_0_less_1 = data[(data['MetabolicDisorderWithoutBMI'] < threshold) & (data['Diabetes_binary'] == 0)].shape[0]
diabetes_1_less_1 = data[(data['MetabolicDisorderWithoutBMI'] < threshold) & (data['Diabetes_binary'] == 1)].shape[0]

diabetes_0_gte_1 = data[(data['MetabolicDisorderWithoutBMI'] >= threshold) & (data['Diabetes_binary'] == 0)].shape[0]
diabetes_1_gte_1 = data[(data['MetabolicDisorderWithoutBMI'] >= threshold) & (data['Diabetes_binary'] == 1)].shape[0]

diabetic = (data['Diabetes_binary'] == 1).sum()  # Counts True values
non_diabetic = (data['Diabetes_binary'] == 0).sum()  # Counts True values

print(f"Diabetic: {diabetic}")
print(f"Non-diabetic: {non_diabetic}")

print(f"Metabolic Disorder < {threshold}: {metabolic_disorder_less_1}")
print(f"Metabolic Disorder >= {threshold}: {metabolic_disorder_gte_1}")
print(f"New Metabolic Disorder < {threshold}: {new_metabolic_disorder_less_1}")
print(f"New Metabolic Disorder >= {threshold}: {new_metabolic_disorder_gte_1}")

print(f"Diabetes = 0 and Metabolic Disorder < {threshold}: {diabetes_0_less_1}")
print(f"Diabetes = 1 and Metabolic Disorder < {threshold}: {diabetes_1_less_1}")
print(f"Diabetes = 0 and Metabolic Disorder >= {threshold}: {diabetes_0_gte_1}")
print(f"Diabetes = 1 and Metabolic Disorder >= {threshold}: {diabetes_1_gte_1}")