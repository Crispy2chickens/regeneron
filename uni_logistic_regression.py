import pandas as pd
import statsmodels.api as sm

# Step 1: Load the data
data = pd.read_csv('diabetes_binary_health_indicators_BRFSS2023.csv')

# Step 2: Set target variable
y = data['Diabetes_binary']

# Step 3: List of independent variables for univariable analysis
independent_variables = [
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
    'Income', 
]

# Step 4: Perform univariable logistic regression for each independent variable
for variable in independent_variables:
    # Step 4a: Define the model
    X = sm.add_constant(data[[variable]])  # Add constant for intercept
    
    # Step 4b: Fit the model
    model = sm.Logit(y, X)
    result = model.fit(disp=0)  # disp=0 to avoid printing fitting messages
    
    # Step 4c: Print the results
    print(f"Results for {variable}:")
    print(result.summary())
    print("\n")
