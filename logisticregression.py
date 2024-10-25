import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

file_path = 'diabetes_binary_final.csv'  
data = pd.read_csv(file_path)

# bmi = data['BMI'].values.reshape(-1, 1)  
metabolicdisorders = data['Stroke'].values.reshape(-1, 1)  
diabetes_outcomes = data['Diabetes_binary'].values      

print(f"Non-diabetic: {sum(diabetes_outcomes == 0)}, Diabetic: {sum(diabetes_outcomes == 1)}")

X_temp, X_test, y_temp, y_test = train_test_split(metabolicdisorders, diabetes_outcomes, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  

model = LogisticRegression(solver='liblinear', class_weight='balanced')
model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.2%}")

y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.2%}")

beta_0 = model.intercept_[0]
beta_1 = model.coef_[0][0]
print(f"Intercept (beta_0): {beta_0:.4f}")
# print(f"Coefficient for BMI (beta_1): {beta_1:.4f}")
print(f"Coefficient for Metabolic Disorders With BMI (beta_1): {beta_1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

y_prob = model.predict_proba(X_test)[:, 1]  
auc = roc_auc_score(y_test, y_prob)
print(f"AUC Score: {auc:.4f}")

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()
