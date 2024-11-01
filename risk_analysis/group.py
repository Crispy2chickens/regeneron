import pandas as pd

# Load the dataset
data = pd.read_csv('new_diabetes_binary_health_indicators_geometric_BRFSS2023.csv')

# Define the threshold for the old risk metric
threshold = 1.41

# Calculate proportions and mean reductions for multiple factors
factors = [
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
    'Income'
]

# Prepare to collect results
results = {}

for factor in factors:
    # Define high and low risk groups based on 'MetabolicDisorderWithoutBMI'
    high_risk_old = data[data['MetabolicDisorderWithoutBMI'] > threshold]
    low_risk_old = data[data['MetabolicDisorderWithoutBMI'] <= threshold]

    # Shifted high-risk and low-risk individuals
    high_risk_old_shifted = high_risk_old[high_risk_old['NewMetabolicDisorder'] <= threshold]
    low_risk_old_shifted = low_risk_old[low_risk_old['NewMetabolicDisorder'] <= threshold]

    # Calculate proportions
    high_risk_proportion = len(high_risk_old_shifted) / len(high_risk_old) if len(high_risk_old) > 0 else 0
    low_risk_proportion = len(low_risk_old_shifted) / len(low_risk_old) if len(low_risk_old) > 0 else 0

    # Calculate mean reductions
    old_high_risk_mean = high_risk_old[factor].mean() if len(high_risk_old) > 0 else 0
    new_high_risk_mean = high_risk_old_shifted[factor].mean() if len(high_risk_old_shifted) > 0 else 0
    mean_reduction = old_high_risk_mean - new_high_risk_mean

    # Store results
    results[factor] = {
        'Proportion of high-risk moving to lower risk': high_risk_proportion,
        'Proportion of low-risk remaining in lower risk': low_risk_proportion,
        'Mean reduction': mean_reduction
    }

# Print the results
for factor, metrics in results.items():
    print(f"--- {factor} ---")
    print(f"Proportion of high-risk individuals moving to lower risk: {metrics['Proportion of high-risk moving to lower risk']:.2f}")
    print(f"Proportion of low-risk individuals remaining in lower risk: {metrics['Proportion of low-risk remaining in lower risk']:.2f}")
    print(f"Mean reduction for high-risk individuals: {metrics['Mean reduction']:.2f}\n")