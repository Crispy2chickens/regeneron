import pandas as pd

# Load your datasets
brfss = pd.read_csv('diabetes_binary_health_indicators_BRFSS2023.csv')
brfss_index = pd.read_csv('diabetes_binary_health_indicators_with_index_BRFSS2023.csv')

# Adjust pandas display options to show all columns
pd.set_option('display.max_columns', None)  # No limit on the number of columns

# Print the first few rows of each DataFrame
print("First few rows of brfss DataFrame:")
print(brfss.head().to_string(index=False))

print("\nFirst few rows of brfss_index DataFrame:")
print(brfss_index.head().to_string(index=False))
