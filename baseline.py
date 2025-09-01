import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report, f1_score
import joblib

df = pd.read_csv('hmda.csv', low_memory=False)

# Target column
target_col = 'action_taken'

features_to_keep = [
    "loan_amount", "property_value", "income",
    "applicant_credit_score_type",
    "tract_population", "tract_minority_population_percent",
    "ffiec_msa_md_median_family_income", "tract_to_msa_income_percentage",
    "derived_ethnicity", "derived_race"
]

X = df[features_to_keep].copy() 
y = df[target_col].copy()

# Remove rows where target is NaN
mask = y.notna()
X = X[mask]
y = y[mask]

num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# --- PRE-PROCESSING ---
print("--- PRE-PROCESSING ---")

# Impute missing values for numeric columns
for col in num_cols:
    X[col] = pd.to_numeric(X[col], errors='coerce')
    X[col] = X[col].fillna(X[col].median())

# Fill missing categorical values and encode them
label_encoders = {}
for col in cat_cols:
    X[col] = X[col].fillna("Unknown")
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

print("Data head after preprocessing:")
print(X.head())
print("\n")

# Create a stratified split for a realistic test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Store the sensitive attributes for auditing purposes from the test set.
sensitive_test = X_test[['derived_ethnicity', 'derived_race']].copy()
sensitive_test['derived_ethnicity'] = label_encoders['derived_ethnicity'].inverse_transform(sensitive_test['derived_ethnicity'])
sensitive_test['derived_race'] = label_encoders['derived_race'].inverse_transform(sensitive_test['derived_race'])


# Initialize and train the Logistic Regression model
model = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced')
model.fit(X_train, y_train)

# Predictions on the realistic test set
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# --- Model Evaluation (Baseline Metrics) ---
print("--- Baseline Model Evaluation ---")
print("Evaluation on Realistic Test Set:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
print("\n")

# --- Demographic Fairness Metrics (Consolidated) ---
print("--- Demographic Fairness Metrics (Consolidated) ---")

# Combine test results with sensitive attributes for easy analysis.
test_results = pd.DataFrame({
    'y_true': y_test,
    'y_pred': y_pred,
    'derived_race': sensitive_test['derived_race'],
    'derived_ethnicity': sensitive_test['derived_ethnicity']
})

# Define the "privileged" group for race and ethnicity
privileged_races = ["White", "Asian"]
privileged_ethnicity = "Not Hispanic or Latino"

# --- Demographic Parity ---
# Measures if the acceptance rate is the same for all groups.
# A value of 1.0 means perfect parity.

# Demographic Parity by Race
is_privileged_race = test_results['derived_race'].isin(privileged_races)
privileged_race_rate = test_results[is_privileged_race]['y_pred'].mean()
unprivileged_race_rate = test_results[~is_privileged_race]['y_pred'].mean()
demographic_parity_race = unprivileged_race_rate / privileged_race_rate if privileged_race_rate != 0 else np.inf
print("Demographic Parity by Race:")
print(f"  Acceptance Rate for Privileged Group ('White', 'Asian'): {privileged_race_rate:.4f}")
print(f"  Acceptance Rate for Unprivileged Groups: {unprivileged_race_rate:.4f}")
print(f"  Demographic Parity Ratio (Unprivileged / Privileged): {demographic_parity_race:.4f}\n")

# --- Equalized Odds ---
# Measures if the model has the same True Positive Rate (Recall) across groups.
# A value of 0.0 means perfect parity.

# Equalized Odds by Race
def calculate_tpr(df_group):
    true_positives = df_group[(df_group['y_true'] == 1) & (df_group['y_pred'] == 1)].shape[0]
    total_positives = df_group[df_group['y_true'] == 1].shape[0]
    return true_positives / total_positives if total_positives > 0 else 0

privileged_race_tpr = calculate_tpr(test_results[is_privileged_race])
unprivileged_race_tpr = calculate_tpr(test_results[~is_privileged_race])

print("Equalized Odds (True Positive Rate) by Race:")
print(f"  TPR for Privileged Group ('White', 'Asian'): {privileged_race_tpr:.4f}")
print(f"  TPR for Unprivileged Groups: {unprivileged_race_tpr:.4f}")
print(f"  TPR Difference (Unprivileged - Privileged): {unprivileged_race_tpr - privileged_race_tpr:.4f}\n")

# --- Poster Graphics & Analysis ---

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Denied (0)', 'Approved (1)'],
            yticklabels=['Denied (0)', 'Approved (1)'])
plt.title('Confusion Matrix for Loan Decisions')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot Feature Importance
plt.figure(figsize=(10, 8))
feature_names = X.columns
coefficients = model.coef_[0]
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': np.abs(coefficients)})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance from Logistic Regression Coefficients')
plt.xlabel('Impact on Prediction (Absolute Coefficient Value)')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
