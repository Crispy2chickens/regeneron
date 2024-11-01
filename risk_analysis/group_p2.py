import pandas as pd

# Load the dataset
data = pd.read_csv('new_diabetes_binary_health_indicators_geometric_BRFSS2023.csv')

# Define the threshold for the old risk metric
threshold = 1.41

# Define BMI groups
def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 24.9:
        return 'Normal weight'
    elif 25.0 <= bmi < 29.9:
        return 'Overweight'
    else:
        return 'Obesity'

# Create a new column for BMI categories
data['BMI_Category'] = data['BMI'].apply(categorize_bmi)

# List of demographic or health-related factors to segment by
demographic_factors = [
    'BMI_Category', 
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
]

# Store results for High_Risk_Shifted
high_risk_shifted_results = {}

for factor in demographic_factors:
    # Group by the demographic factor and calculate High_Risk_Shifted
    grouped_data = data.groupby(factor).apply(lambda x: (
        ((x['MetabolicDisorderWithoutBMI'] > threshold) & (x['NewMetabolicDisorder'] <= threshold)).sum()
    )).reset_index(name='High_Risk_Shifted')  # Explicitly name the resulting column

    high_risk_shifted_results[factor] = grouped_data

# Print only the High_Risk_Shifted results
for factor, results in high_risk_shifted_results.items():
    print(f"--- {factor} ---")
    print(results[['High_Risk_Shifted']])
