# Get Data

import os
import pandas as pd
import numpy as np
import random
random.seed(1)

brfss_2023_dataset = pd.read_csv('LLCP2023.csv')

# print(brfss_2023_dataset.shape)
# (433323, 350)

# pd.set_option('display.max_columns', 500)
# print(brfss_2023_dataset.head())

brfss_df_selected = brfss_2023_dataset[['DIABETE4',
                                         '_RFHYPE6',  
                                         'TOLDHI3',  
                                         '_BMI5', 'WEIGHT2',
                                         'SMOKE100', 
                                         'CVDSTRK3', '_MICHD', 
                                         '_TOTINDA', 
                                         '_RFDRHV8', 
                                         '_HLTHPL1', 'MEDCOST1', 
                                         'GENHLTH', 'PHYSHLTH', 'MENTHLTH', 'POORHLTH', 'DIFFWALK', 
                                         '_SEX', '_AGEG5YR', '_IMPRACE', '_EDUCAG', 'INCOME3' ]]

# print(brfss_df_selected.shape)
# (433323, 22)

# print(brfss_df_selected.head())


# 2 Clean data
brfss_df_selected = brfss_df_selected.dropna()
# print(brfss_df_selected.shape)
# (195702, 22)

# DIABETE4
# going to make this ordinal. 0 is for no diabetes or only during pregnancy, 1 is for pre-diabetes or borderline diabetes, 2 is for yes diabetes
# Remove all 7 (dont knows)
# Remove all 9 (refused)
brfss_df_selected['DIABETE4'] = brfss_df_selected['DIABETE4'].replace({2:0, 3:0, 1:2, 4:1})
brfss_df_selected = brfss_df_selected[brfss_df_selected.DIABETE4 != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.DIABETE4 != 9]
# print(brfss_df_selected.DIABETE4.unique())
# [0. 2. 1.]

# _RFHYPE6
#Change 1 to 0 so it represetnts No high blood pressure and 2 to 1 so it represents high blood pressure
brfss_df_selected['_RFHYPE6'] = brfss_df_selected['_RFHYPE6'].replace({1:0, 2:1})
brfss_df_selected = brfss_df_selected[brfss_df_selected._RFHYPE6 != 9]
# print(brfss_df_selected._RFHYPE6.unique())
# [1. 0.]

# TOLDHI3
# Change 2 to 0 because it is No
# Remove all 7 (dont knows)
# Remove all 9 (refused)
brfss_df_selected['TOLDHI3'] = brfss_df_selected['TOLDHI3'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.TOLDHI3 != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.TOLDHI3 != 9]
# print(brfss_df_selected.TOLDHI3.unique())
# [1. 0.]

#_BMI5 (no changes, just note that these are BMI * 100. So for example a BMI of 4018 is really 40.18)
brfss_df_selected['_BMI5'] = brfss_df_selected['_BMI5'].div(100).round(0)
# print(brfss_df_selected._BMI5.unique())

# WEIGHT2
brfss_df_selected = brfss_df_selected[brfss_df_selected.WEIGHT2 != 7777]
brfss_df_selected = brfss_df_selected[brfss_df_selected.WEIGHT2 != 9999]
brfss_df_selected.loc[
    (brfss_df_selected['WEIGHT2'] >= 50) & (brfss_df_selected['WEIGHT2'] <= 776), 
    'WEIGHT2'
] = round(brfss_df_selected['WEIGHT2'] / 2.20462)
brfss_df_selected.loc[
    (brfss_df_selected['WEIGHT2'] >= 9023) & (brfss_df_selected['WEIGHT2'] <= 9352), 
    'WEIGHT2'
] = brfss_df_selected['WEIGHT2'] - 9000
brfss_df_selected.rename(columns={'WEIGHT2': 'WEIGHT2_KG'}, inplace=True)
# print(brfss_df_selected['WEIGHT2_KG'].unique())

#5 SMOKE100
# Change 2 to 0 because it is No
# Remove all 7 (dont knows)
# Remove all 9 (refused)
brfss_df_selected['SMOKE100'] = brfss_df_selected['SMOKE100'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.SMOKE100 != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.SMOKE100 != 9]
# print(brfss_df_selected.SMOKE100.unique())
# [1. 0.]

# CVDSTRK3]# Change 2 to 0 because it is No
# Remove all 7 (dont knows)
# Remove all 9 (refused)
brfss_df_selected['CVDSTRK3'] = brfss_df_selected['CVDSTRK3'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.CVDSTRK3 != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.CVDSTRK3 != 9]
# print(brfss_df_selected.CVDSTRK3.unique())
# [0. 1.]

# _MICHD
#Change 2 to 0 because this means did not have MI or CHD
brfss_df_selected['_MICHD'] = brfss_df_selected['_MICHD'].replace({2: 0})
# print(brfss_df_selected._MICHD.unique())
# [0. 1.]

# _TOTINDA
# 1 for physical activity
# change 2 to 0 for no physical activity
# Remove all 9 (don't know/refused)
brfss_df_selected['_TOTINDA'] = brfss_df_selected['_TOTINDA'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected._TOTINDA != 9]
# print(brfss_df_selected._TOTINDA.unique())
# [1. 0.]

# _RFDRHV8
# Change 1 to 0 (1 was no for heavy drinking). change all 2 to 1 (2 was yes for heavy drinking)
# remove all dont knows and missing 9
brfss_df_selected['_RFDRHV8'] = brfss_df_selected['_RFDRHV8'].replace({1:0, 2:1})
brfss_df_selected = brfss_df_selected[brfss_df_selected._RFDRHV8 != 9]
# print(brfss_df_selected._RFDRHV8.unique())
# [0. 1.]

# _HLTHPL1
# 1 is yes, change 2 to 0 because it is No health care access
# remove 9 for don't know or refused
brfss_df_selected['_HLTHPL1'] = brfss_df_selected['_HLTHPL1'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected._HLTHPL1 != 9]
# print(brfss_df_selected._HLTHPL1.unique())
# [1. 0.]

# MEDCOST1
# Change 2 to 0 for no, 1 is already yes
# remove 7 for don/t know and 9 for refused
brfss_df_selected['MEDCOST1'] = brfss_df_selected['MEDCOST1'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.MEDCOST1 != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.MEDCOST1 != 9]
# print(brfss_df_selected.MEDCOST1.unique())
# [1. 0.]

# GENHLTH
# This is an ordinal variable that I want to keep (1 is Excellent -> 5 is Poor)
# Remove 7 and 9 for don't know and refused
brfss_df_selected = brfss_df_selected[brfss_df_selected.GENHLTH != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.GENHLTH != 9]
# print(brfss_df_selected.GENHLTH.unique())
# [4. 2. 3. 5. 1.]

# PHYSHLTH
# already in days so keep that, scale will be 0-30
# change 88 to 0 because it means none (no bad mental health days)
# remove 77 and 99 for don't know not sure and refused
brfss_df_selected['PHYSHLTH'] = brfss_df_selected['PHYSHLTH'].replace({88:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.PHYSHLTH != 77]
brfss_df_selected = brfss_df_selected[brfss_df_selected.PHYSHLTH != 99]
# print(brfss_df_selected.PHYSHLTH.unique())

# MENTHLTH
# already in days so keep that, scale will be 0-30
# change 88 to 0 because it means none (no bad mental health days)
# remove 77 and 99 for don't know not sure and refused
brfss_df_selected['MENTHLTH'] = brfss_df_selected['MENTHLTH'].replace({88:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.MENTHLTH != 77]
brfss_df_selected = brfss_df_selected[brfss_df_selected.MENTHLTH != 99]
# print(brfss_df_selected.MENTHLTH.unique())

# POORHLTH
# already in days so keep that, scale will be 0-30
# change 88 to 0 because it means none (no bad mental health days)
# remove 77 and 99 for don't know not sure and refused
brfss_df_selected['POORHLTH'] = brfss_df_selected['POORHLTH'].replace({88:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.POORHLTH != 77]
brfss_df_selected = brfss_df_selected[brfss_df_selected.POORHLTH != 99]
# print(brfss_df_selected.POORHLTH.unique())

# DIFFWALK
# change 2 to 0 for no. 1 is already yes
# remove 7 and 9 for don't know not sure and refused
brfss_df_selected['DIFFWALK'] = brfss_df_selected['DIFFWALK'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.DIFFWALK != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.DIFFWALK != 9]
# print(brfss_df_selected.DIFFWALK.unique())
# [1. 0.]

# _SEX
# in other words - is respondent male (somewhat arbitrarily chose this change because men are at higher risk for heart disease)
# change 2 to 0 (female as 0). Male is 1
brfss_df_selected['_SEX'] = brfss_df_selected['_SEX'].replace({2:0})
# print(brfss_df_selected._SEX.unique())
# [0. 1.]

# _AGEG5YR
# already ordinal. 1 is 18-24 all the way up to 13 wis 80 and older. 5 year increments.
# remove 14 because it is don't know or missing
brfss_df_selected = brfss_df_selected[brfss_df_selected._AGEG5YR != 14]
# print(brfss_df_selected._AGEG5YR.unique())

# _IMPRACE
# Define the ordinal mapping based on assumed diabetes risk or prevalence
race_mapping = {
    1: 1,  # White, Non-Hispanic
    3: 2,  # Asian, Non-Hispanic
    6: 3,  # Other race, Non-Hispanic
    5: 4,  # Hispanic
    2: 5,  # Black, Non-Hispanic
    4: 6   # American Indian/Alaskan Native, Non-Hispanic
}
brfss_df_selected['_IMPRACE'] = brfss_df_selected['_IMPRACE'].replace(race_mapping)
# print(brfss_df_selected._IMPRACE.unique())

# _EDUCAG
# This is already an ordinal variable with 1 being never attended school or kindergarten only up to 6 being college 4 years or more
# Scale here is 1-6
# Remove 9 for refused:
brfss_df_selected = brfss_df_selected[brfss_df_selected._EDUCAG != 9]
# print(brfss_df_selected._EDUCAG.unique())

# INCOME3
# Variable is already ordinal 
# Remove 77 and 99 for don't know and refused
brfss_df_selected = brfss_df_selected[brfss_df_selected.INCOME3 != 77]
brfss_df_selected = brfss_df_selected[brfss_df_selected.INCOME3 != 99]
# print(brfss_df_selected.INCOME3.unique())

# print(brfss_df_selected.shape)
# (148821, 22)

# print(brfss_df_selected.head())

# print(brfss_df_selected.groupby(['DIABETE4']).size())
# 0.0    121258
# 1.0      3985
# 2.0     23578

brfss = brfss_df_selected.rename(columns = {'DIABETE4':'Diabetes_012',
                                         '_RFHYPE6':'HighBP',  
                                         'TOLDHI3':'HighChol',  
                                         '_BMI5':'BMI', 'WEIGHT2_KG':'Weight',
                                         'SMOKE100':'Smoker', 
                                         'CVDSTRK3':'Stroke', '_MICHD':'HeartDiseaseorAttack', 
                                         '_TOTINDA':'PhysActivity', 
                                         '_RFDRHV8':'HvyAlcoholConsump', 
                                         '_HLTHPL1':'AnyHealthcare', 'MEDCOST1':'NoDocbcCost', 
                                         'GENHLTH':'GenHlth', 'PHYSHLTH':'PhysHlth', 'MENTHLTH':'MentHlth', 'POORHLTH':'PoorHlth', 'DIFFWALK':'DiffWalk', 
                                         '_SEX':'Sex', '_AGEG5YR':'AgeGroup', '_IMPRACE':'Race', '_EDUCAG':'Education', 'INCOME3':'Income'})

# print(brfss.head())
# print(brfss.shape)
# (148821, 22)

# print(brfss.groupby(['Diabetes_012']).size())
# 0.0    121258
# 1.0      3985
# 2.0     23578


# Save to CSV
# brfss.to_csv('diabetes_012_health_indicators_BRFSS2023.csv', sep=",", index=False)

# Create a deep copy of the DataFrame
brfss_binary = brfss.copy()

# Replace values in the existing Diabetes_012 column
brfss_binary['Diabetes_012'] = brfss_binary['Diabetes_012'].replace({0: 0, 1: 1, 2: 1})

# Rename the column to Diabetes_binary
brfss_binary = brfss_binary.rename(columns={'Diabetes_012': 'Diabetes_binary'})

# def calculate_metalobic_with_BMI(row):
#     # Assign BMI score based on ranges
#     if row['BMI'] < 18.5:
#         bmi_score = 0
#     elif 18.5 <= row['BMI'] < 25:
#         bmi_score = 0
#     elif 25 <= row['BMI'] < 30:
#         bmi_score = 1
#     else:  # BMI >= 30
#         bmi_score = 1
    
#     risk_score = (
#         bmi_score + 
#         row['HighBP'] + 
#         row['HighChol'] + 
#         row['HeartDiseaseorAttack']
#     )
    
#     return risk_score

def calculate_metalobic_without_BMI(row):
    risk_score = (
        row['HighBP'] + 
        row['HighChol'] + 
        row['HeartDiseaseorAttack']+
        row['Stroke']
    )
    
    return risk_score

def calculate_metalobic_with_BMI(row):
    # Assign BMI score based on ranges
    if row['BMI'] < 18.5:
        bmi_score = 1  # Low risk for underweight
    elif 18.5 <= row['BMI'] < 25:
        bmi_score = 1  # Normal weight contributes less risk
    elif 25 <= row['BMI'] < 30:
        bmi_score = 2  # Overweight contributes moderate risk
    else:  # BMI >= 30
        bmi_score = 3  # Obese contributes high risk
    
    risk_score = (
        bmi_score + 
        row['HighBP'] + 
        row['HighChol'] + 
        row['HeartDiseaseorAttack'] +
        row['Stroke']
    )
    
    return risk_score


# def calculate_metalobic_with_BMI_multiplicative(row):
#     # Assign BMI score based on ranges
#     if row['BMI'] < 18.5:
#         bmi_score = 0
#     elif 18.5 <= row['BMI'] < 25:
#         bmi_score = 1  # Normal weight contributes less risk
#     elif 25 <= row['BMI'] < 30:
#         bmi_score = 2  # Overweight contributes moderate risk
#     else:  # BMI >= 30
#         bmi_score = 3  # Obese contributes high risk
    
#     # Ensure that the scores are non-negative
#     highBP = row['HighBP'] if row['HighBP'] > 0 else 1  # Avoid multiplying by zero
#     highChol = row['HighChol'] if row['HighChol'] > 0 else 1
#     heartDisease = row['HeartDiseaseorAttack'] if row['HeartDiseaseorAttack'] > 0 else 1
    
#     risk_score = (
#         bmi_score * 
#         highBP * 
#         highChol * 
#         heartDisease
#     )
    
#     return risk_score

# def calculate_metalobic_without_BMI_multiplicative(row):
#     highBP = row['HighBP'] if row['HighBP'] > 0 else 1  # Avoid multiplying by zero
#     highChol = row['HighChol'] if row['HighChol'] > 0 else 1
#     heartDisease = row['HeartDiseaseorAttack'] if row['HeartDiseaseorAttack'] > 0 else 1

#     risk_score = (
#         highBP * 
#         highChol * 
#         heartDisease
#     )
    
#     return risk_score

# def calculate_metalobic_with_BMI_geometric(row):
#     # Assign BMI score based on ranges
#     if row['BMI'] < 18.5:
#         bmi_score = 1  # Low risk for underweight
#     elif 18.5 <= row['BMI'] < 25:
#         bmi_score = 1  # Normal weight contributes less risk
#     elif 25 <= row['BMI'] < 30:
#         bmi_score = 2  # Overweight contributes moderate risk
#     else:  # BMI >= 30
#         bmi_score = 3  # Obese contributes high risk

#     # Ensure non-negative scores
#     highBP = row['HighBP'] if row['HighBP'] > 0 else 1
#     highChol = row['HighChol'] if row['HighChol'] > 0 else 1
#     heartDisease = row['HeartDiseaseorAttack'] if row['HeartDiseaseorAttack'] > 0 else 1
    
#     # Calculate the geometric mean
#     risk_score = np.exp(np.log(bmi_score) + np.log(highBP) + np.log(highChol) + np.log(heartDisease)) / 4
    
#     return risk_score

# def calculate_metalobic_without_BMI_geometric(row):
#     # Ensure non-negative scores
#     highBP = row['HighBP'] if row['HighBP'] > 0 else 1
#     highChol = row['HighChol'] if row['HighChol'] > 0 else 1
#     heartDisease = row['HeartDiseaseorAttack'] if row['HeartDiseaseorAttack'] > 0 else 1
    
#     # Calculate the geometric mean
#     risk_score = np.exp(np.log(highBP) + np.log(highChol) + np.log(heartDisease)) / 3
    
#     return risk_score

# Apply the function to create a new column 'Diabetes_Risk_Index'
brfss_binary['MetabolicDisorderWithBMI'] = brfss_binary.apply(calculate_metalobic_with_BMI, axis=1)
brfss_binary['MetabolicDisorderWithoutBMI'] = brfss_binary.apply(calculate_metalobic_without_BMI, axis=1)

# Get the cases for 1 (diabetic and prediabetic)
is1 = brfss_binary['Diabetes_binary'] == 1
brfss_5050_1 = brfss_binary[is1]

# Get the cases for 0 (non-diabetes)
is0 = brfss_binary['Diabetes_binary'] == 0
brfss_5050_0 = brfss_binary[is0] 

# Select random cases from the 0 (non-diabetes) group
brfss_5050_0_rand1 = brfss_5050_0.sample(n=len(brfss_5050_1), random_state=42)

# Append the selected 0s to the 1s
brfss_5050 = brfss_5050_0_rand1._append(brfss_5050_1, ignore_index=True)

# Display the first and last few rows
# print(brfss_5050.head())
# print(brfss_5050.tail())

# # Check the size of each group
# print(brfss_5050.groupby(['Diabetes_binary']).size())

# brfss_5050.to_csv('diabetes_binary_health_indicators_BRFSS2023.csv', sep=",", index=False)
# brfss_5050.to_csv('diabetes_binary_health_indicators_with_index_BRFSS2023.csv', sep=",", index=False)
# brfss_5050.to_csv('diabetes_binary_multiplicative_BRFSS2023.csv', sep=",", index=False)
# brfss_5050.to_csv('diabetes_binary_new_geometric_BRFSS2023.csv', sep=",", index=False)
# brfss_5050.to_csv('diabetes_binary_additive_BRFSS2023.csv', sep=",", index=False)
brfss_5050.to_csv('diabetes_binary_final.csv', sep=",", index=False)