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
                                         '_BMI5', 
                                         'CVDSTRK3', '_MICHD', 
                                         '_TOTINDA', 
                                         '_HLTHPL1',
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

# _HLTHPL1
# 1 is yes, change 2 to 0 because it is No health care access
# remove 9 for don't know or refused
brfss_df_selected['_HLTHPL1'] = brfss_df_selected['_HLTHPL1'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected._HLTHPL1 != 9]
# print(brfss_df_selected._HLTHPL1.unique())
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
    1: 1,  # White, Non-Hispanic (lower-risk baseline)
    3: 2,  # Asian, Non-Hispanic (moderate-risk)
    6: 2,  # Other race, Non-Hispanic (moderate-risk, includes Pacific Islander)
    5: 3,  # Hispanic (higher-risk)
    2: 3,  # Black, Non-Hispanic (higher-risk)
    4: 3   # American Indian/Alaskan Native, Non-Hispanic (higher-risk)
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
# # (279926, 13)

# print(brfss_df_selected.head())

# print(brfss_df_selected.groupby(['DIABETE4']).size())
# 0.0    231460
# 1.0      7152
# 2.0     41314

brfss = brfss_df_selected.rename(columns = {'DIABETE4':'Diabetes_012',
                                         '_RFHYPE6':'HighBP',  
                                         'TOLDHI3':'HighChol',  
                                         '_BMI5':'BMI', 
                                         'CVDSTRK3':'Stroke', '_MICHD':'HeartDiseaseorAttack', 
                                         '_TOTINDA':'PhysActivity', 
                                         '_HLTHPL1':'AnyHealthcare', 
                                         '_SEX':'Sex', '_AGEG5YR':'AgeGroup', '_IMPRACE':'Race', '_EDUCAG':'Education', 'INCOME3':'Income'})

# print(brfss.head())
# print(brfss.shape)
# (279926, 13)

# print(brfss.groupby(['Diabetes_012']).size())
# 0.0    231460
# 1.0      7152
# 2.0     41314


# Save to CSV
# brfss.to_csv('diabetes_012_health_indicators_BRFSS2023.csv', sep=",", index=False)

# Define overweight and obesity criteria based on BMI
overweight_criteria = (brfss['BMI'] >= 25) & (brfss['BMI'] < 30)
obese_criteria = brfss['BMI'] >= 30

# Define additional health risks and diabetes condition
health_risks = (brfss['HighBP'] == 1) | (brfss['HighChol'] == 1) | (brfss['Stroke'] == 1) | (brfss['HeartDiseaseorAttack'] == 1)
diabetes_condition = brfss['Diabetes_012'] == 2

# Create the "at risk?" column
brfss['AtRisk'] = ((obese_criteria) | (overweight_criteria & health_risks) | (diabetes_condition)).astype(int)

print(brfss.head())

def calculate_metalobic_geometric_without_BMI(row):
    risk_score = (
        (1+row['HighBP']) * (1+row['HighChol']) * (1+row['HeartDiseaseorAttack']) * (1+(row['Stroke']))
    ) ** (1/4)
    
    return risk_score

def calculate_new_metabolic_geometric_without_BMI(row):
    new_risk_score = (
        (1+row['HighBP'] * 0.9789) *  
        (1+row['HighChol'] * 0.958) *  
        (1+row['HeartDiseaseorAttack'] * 0.80) *  
        (1+row['Stroke'] * 0.80)  
    ) ** (1/4)  

    return new_risk_score

# # Apply the function to create a new column 'Diabetes_Risk_Index'
brfss['MetabolicDisorderWithoutBMI'] = brfss.apply(calculate_metalobic_geometric_without_BMI, axis=1)
brfss['NewMetabolicDisorder'] = brfss.apply(calculate_new_metabolic_geometric_without_BMI, axis=1)

print(brfss.head())

brfss.to_csv('current_diabetes_binary_health_indicators_geometric_BRFSS2023.csv', sep=",", index=False)