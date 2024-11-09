import pandas as pd

# Load the dataset
data = pd.read_csv('current_diabetes_binary_health_indicators_geometric_BRFSS2023.csv')

threshold = 1.41

# Filter the data based on the conditions
filtered_data = data[(data['MetabolicDisorderWithoutBMI'] > threshold) & 
                     ((data['HighBP'] == 1) | (data['HighChol'] == 1) | (data['AgeGroup'] > 4))]

# Get the number of people matching the criteria
count = filtered_data.shape[0]

# Print the result
print(f"Number of people above threshold and (HighBP=1 OR HighChol=1 OR AgeGroup > 4): {count}")


#selective
filtered_data_2 = data[(data['MetabolicDisorderWithoutBMI'] > threshold) & 
                     (((data['HighBP'] == 1) & (data['HighChol'] == 1)) & (data['AgeGroup'] > 6))]

# Get the number of people matching the criteria
count2 = filtered_data_2.shape[0]

# Print the result
print(f"Number of people above threshold and ((HighBP=1 AND HighChol=1) AND AgeGroup > 6): {count2}")



#limited
filtered_data_3 = data[(data['MetabolicDisorderWithoutBMI'] > threshold) &
                     ((data['HighBP'] == 1) | (data['HighChol'] == 1) | (data['AgeGroup'] > 6))
                     & ((data['HeartDiseaseorAttack'] == 1) | (data['Stroke'] == 1))]

# Get the number of people matching the criteria
count3 = filtered_data_3.shape[0]

# Print the result
print(f"Number of people above threshold and ((HighBP=1 AND HighChol=1) AND AgeGroup > 6): {count3}")


# obese
filtered_data_4 = data[(data['Diabetes_012'] == 2) | 
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
        )]
count4 = filtered_data_4.shape[0]
print(f"Current Criteria: {count4}")