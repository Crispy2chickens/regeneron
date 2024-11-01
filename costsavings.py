import numpy as np
import matplotlib.pyplot as plt

# Parameters
annual_risk_reduction = 0.306  # 0.5% risk reduction annually
years = np.arange(0, 21)  # Every year up to 20 years
initial_population = 136e6  # Initial at-risk population
diabetes_cost_per_patient = 10752  # Annual cost per diabetes patient
glp1_cost_per_patient = 1142.5  # Annual cost of GLP-1 drugs

# Decay factor for diabetes risk reduction
risk_reduction_rate = 1 - annual_risk_reduction

# Calculate population at risk over time
population_at_risk = initial_population * (risk_reduction_rate ** years)

# Calculate the number of diabetes cases avoided each year
diabetes_cases_avoided = initial_population - population_at_risk
print(diabetes_cases_avoided)

# Calculate total annual savings from reduced diabetes cases
savings = diabetes_cases_avoided * diabetes_cost_per_patient  # Savings each year

# Calculate the GLP-1 drug cost over the years
glp1_cost = population_at_risk * glp1_cost_per_patient

# Plot
plt.figure(figsize=(10, 6))
plt.plot(years, savings / 1e9, label='Cost-savings from reduced diabetes cases', color='blue')
plt.plot(years, glp1_cost / 1e9, label='Cost of GLP-1 drugs', color='orange')

# Labels and legend
plt.title('Cost-savings from GLP-1 Drugs vs Cost of GLP-1 Drugs')
plt.xlabel('Years')
plt.ylabel('Annual Cost (in billions)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
