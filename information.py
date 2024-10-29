import pandas as pd

df = pd.read_csv('new_diabetes_binary_health_indicators_geometric_BRFSS2023.csv')


# filtered_df = df[df['Diabetes_binary'] == 1]
# average_metric = filtered_df['MetabolicDisorderWithoutBMI'].mean()
# print("Average of the MetabolicDisorderWithoutBMI where diabetes = 1:", average_metric)


# count_high_bp = df[df['HighBP'] == 1].shape[0]
# total_count = df.shape[0]
# percentage_high_bp = (count_high_bp / total_count) * 100 if total_count > 0 else 0

# print("Percentage of people with HighBP = 1:", percentage_high_bp)

# average_BP = df['HighBP'].mean()
# print(("Average of the HighBP:", average_BP))

# average_chol = df['HighChol'].mean()
# print(("Average of the HighChol:", average_chol))

# average_HeartDiseaseorAttack = df['HeartDiseaseorAttack'].mean()
# print(("Average of the HeartDiseaseorAttack:", average_HeartDiseaseorAttack))

# average_stroke = df['Stroke'].mean()
# print(("Average of the Stroke:", average_stroke))

average_old = df['MetabolicDisorderWithoutBMI'].mean()
print(("Average of Old Index:", average_old))

average_new = df['NewMetabolicDisorder'].mean()
print(("Average of New Index:", average_new))

reduction=average_old-average_new
print(reduction)
percentageReduction = reduction/average_old*100
print("Percentage Reduction: ", percentageReduction)

oldOR = 48.414163

ORdifference = oldOR*reduction

print(ORdifference)

print(oldOR-ORdifference)