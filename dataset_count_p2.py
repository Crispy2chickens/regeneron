import pandas as pd

# Load the dataset
data = pd.read_csv('current_diabetes_binary_health_indicators_geometric_BRFSS2023.csv')

# Define the threshold for the old risk metric
threshold = 1.41

# risk_transition = data[((data['HighBP'] == 1) |
#                         (data['HighChol'] == 1) |
#                         (data['AgeGroup'] > 4)) & 
#                         (data['MetabolicDisorderWithoutBMI'] > threshold) & 
#                         (data['NewMetabolicDisorder'] <= threshold)]

# # Count the number of individuals with HighBP, AgeGroup > 4 who moved to low risk
# risk_transition_count = risk_transition.shape[0]

# # Print the results
# print(f"People who moved from high risk to low risk: {risk_transition_count}")


# # selective
# risk_transition_2 = data[
#                      ((data['HighBP'] == 1) | (data['HighChol'] == 1) | (data['AgeGroup'] > 6))
#                      & ((data['HeartDiseaseorAttack'] == 1) | (data['Stroke'] == 1)) &
#                         (data['MetabolicDisorderWithoutBMI'] > threshold) & 
#                         (data['NewMetabolicDisorder'] <= threshold)]

# risk_transition_count_2 = risk_transition_2.shape[0]

# print(f"People age>40 and highBP or highchol who moved from high risk to low risk: {risk_transition_count_2}")


# current
risk_transition_3 = risk_transition_3 = data[
    (
        (data['Diabetes_012'] == 2) | 
        (data["BMI"] >= 30) | 
        (
            (data["BMI"] >= 25) & 
            (data["BMI"] < 30) & 
            (
                (data["HighBP"] == 1) | 
                (data["HighChol"] == 1) | 
                (data["Stroke"] == 1) | 
                (data["HeartDiseaseorAttack"] == 1)
            )
        )
    ) & 
    (data['MetabolicDisorderWithoutBMI'] > threshold) & 
    (data['NewMetabolicDisorder'] <= threshold)
]

risk_transition_count_3 = risk_transition_3.shape[0]

print(f"Current: {risk_transition_count_3}")