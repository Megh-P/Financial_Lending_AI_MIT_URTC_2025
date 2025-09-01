# Import necessary libraries for data manipulation, modeling, and plotting.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix, roc_auc_score
import json
import sys


# --- 1. Data Loading and Preprocessing ---
# This section prepares the data for the hybrid framework.

# Load the dataset
try:
    df = pd.read_csv('hmda.csv', low_memory=False)
except FileNotFoundError:
    print("Error: hmda_baseline.csv not found. Please ensure the file is in the same directory.")
    sys.exit()

target_col = 'action_taken'

# Define MUTABLE features for actionable counterfactuals.
mutable_features = [
    "loan_amount", "property_value", "income",
    "applicant_credit_score_type"
]

fairer_features = mutable_features + [
    "tract_population", "tract_to_msa_income_percentage", "ffiec_msa_md_median_family_income"
    , "tract_minority_population_percent"
]

# Sensitive attributes used ONLY for fairness auditing.
sensitive_features = ['derived_race', 'derived_ethnicity']

# All features for the model training
all_model_features = fairer_features + sensitive_features

# Separate features (X) and target (y).
X = df[fairer_features].copy()
y = df[target_col].copy()

# Store sensitive attributes separately for fairness evaluation.
sensitive_attributes = df[sensitive_features].copy()

# Drop rows with missing target and filter out invalid 'action_taken' values.
mask = y.notna() & y.isin([0, 1])
X = X[mask]
y = y[mask]
sensitive_attributes = sensitive_attributes[mask]

# Define numeric and categorical columns for training data.
numeric_cols = [
    "loan_amount", "property_value", "income",
    "tract_population", "tract_to_msa_income_percentage", "ffiec_msa_md_median_family_income"
    , "tract_minority_population_percent"
]
categorical_cols = ["applicant_credit_score_type"]

# Convert numeric columns and fill missing values.
for col in numeric_cols:
    X[col] = pd.to_numeric(X[col], errors='coerce')
    X[col] = X[col].fillna(X[col].median())

# Fill missing categorical values and encode.
label_encoders = {}
for col in categorical_cols:
    X[col] = X[col].fillna("Unknown")
    le = LabelEncoder()
    # Ensure all values are strings before fitting
    X[col] = le.fit_transform(X[col].astype(str)).astype(int)
    label_encoders[col] = le

print("--- Preprocessing Complete ---")
print(X.head(), "\n")

# Stratified split for main data and sensitive attributes.
X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
    X, y, sensitive_attributes, test_size=0.2, random_state=42, stratify=y
)

# --- 2. Train the Fairer LightGBM Model with a Fairness-Aware Objective ---
print("Training the Fairer LightGBM Model...")

model_dicr = lgb.LGBMClassifier(random_state=42, class_weight='balanced', objective='binary', learning_rate=0.03, n_estimators=100, num_leaves=31, min_child_samples=20)
model_dicr.fit(X_train, y_train)

y_pred_dicr = model_dicr.predict(X_test)
y_proba_dicr = model_dicr.predict_proba(X_test)[:, 1]

# --- 3. Fairer Model Performance Metrics ---
print("\n--- Fairer Model Performance Metrics ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dicr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_dicr, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_dicr):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_dicr):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba_dicr):.4f}")

# --- 4. Counterfactual Generation Function (Modified for LightGBM) ---
def get_minimal_counterfactual(
    model, original_profile, mutable_features, X_train,
    group=None, max_iter=100, step=10
):
    import itertools

    counterfactual_profile = original_profile.copy()
    original_pred = model.predict(original_profile.to_frame().T)[0]

    if original_pred != 0:
        return None, None, None, "Not a denied applicant."

    min_change_value = float('inf')
    best_counterfactual = None
    best_features = None
    best_changes = None
    
    def proportional_cost(feature, old_val, new_val, income):
        if feature == "income":
            return abs(new_val - old_val) / max(income, 1e-6)
        else:
            return abs(new_val - old_val) / max(abs(old_val), 1e-6)

    for r in [1, 2]:
        for feature_combo in itertools.combinations(mutable_features, r):
            temp_profile = original_profile.copy()
            total_cost = 0
            changes = {}

            for feature in feature_combo:
                if feature in ["loan_amount", "property_value", "income"]:
                    current_value = temp_profile[feature]
                    new_value = current_value
                    for _ in range(max_iter):
                        # Use a small step for better granularity
                        new_value += step
                        temp_profile[feature] = new_value
                        if model.predict(temp_profile.to_frame().T)[0] == 1:
                            cost = proportional_cost(feature, current_value, new_value, original_profile['income'])
                            total_cost += cost
                            changes[feature] = new_value - current_value
                            break

            if model.predict(temp_profile.to_frame().T)[0] == 1:
                if total_cost < min_change_value:
                    min_change_value = total_cost
                    best_counterfactual = temp_profile.copy()
                    best_features = feature_combo
                    best_changes = changes

    return (
        best_counterfactual,
        best_features,
        min_change_value if min_change_value != float('inf') else None, #changed this
        "Counterfactual found." if best_counterfactual is not None else "No counterfactual found."
    )

# --- 5. Main Fairness & Recourse Metrics Framework ---
print("\n--- Fairness and Recourse Analysis Framework ---")
print("Analyzing the Fairer model (without protected attributes in training).\n")

privileged_races = ["White", "Asian"]

def get_metrics_for_model(df_results, model_name, denied_sample, mutable_features, X_train):

    # --- Outcome-based Fairness Metrics ---
    print("--- A. Outcome-based Fairness ---")

    print("1. Demographic Parity (Approval Rate by Group):")
    race_approval_rates = df_results.groupby('race')['y_pred'].mean()

    privileged_race_rate = race_approval_rates.loc[race_approval_rates.index.intersection(privileged_races)].mean()
    unprivileged_race_rate = race_approval_rates.loc[~race_approval_rates.index.isin(privileged_races)].mean()
    race_parity_ratio = unprivileged_race_rate / privileged_race_rate if privileged_race_rate > 0 else np.nan
    print(f"    Demographic Parity Ratio (Unprivileged/Privileged) by Race: {race_parity_ratio:.4f}\n")

    print("2. Equalized Odds (True Positive Rate & False Positive Rate):")
    def calculate_group_metrics(df_group):
        counts = confusion_matrix(df_group['y_true'], df_group['y_pred'], labels=[0, 1])
        if counts.shape == (2, 2):
            # Unpack all four values from the flattened confusion matrix
            tn, fp, fn, tp = counts.ravel()
        else:
            # This handles cases where a group has no positive or negative labels
            # The correct way to handle a non-2x2 matrix is to set all counts to 0
            tn = fp = fn = tp = 0
        
        # Now you can calculate TPR
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return pd.Series({'TPR': tpr})

    df_results['Group'] = df_results['race'].apply(lambda r: 'Privileged' if (r == 'White' or r == 'Asian') else ('Unprivileged' if r != 'Race Not Available' else None))    
    group_metrics = df_results.groupby('Group').apply(calculate_group_metrics, include_groups=False)
    
    print(" By Privileged vs. Unprivileged Group:\n", group_metrics, "\n")

    # --- Recourse-based Fairness Metrics ---
    print("\n--- B. Recourse-based Fairness ---")
    denied_results_df = df_results.loc[denied_sample.index].copy()
    denied_results_df['recourse_found'] = False
    denied_results_df['cost'] = np.nan

    print(f"Analyzing {len(denied_sample)} denied applicants for recourse metrics.")
    if not denied_sample.empty:
        total_applicants = len(denied_sample)
        for i, (index, denied_profile) in enumerate(denied_sample.iterrows()):
            print(f"Progress: {i + 1}/{total_applicants} analyzed...", end='\r')
            _, _, change_value, _ = get_minimal_counterfactual(
                model_dicr, denied_profile[fairer_features], mutable_features, X_train
            )
            if change_value is not None:
                denied_results_df.loc[index, 'recourse_found'] = True
                cost_value = np.nan
                if isinstance(change_value, (pd.Series, np.ndarray, list)):
                    try:
                        cost_value = float(np.ravel(change_value)[0])
                    except Exception:
                        cost_value = np.nan
                elif isinstance(change_value, dict):
                    # Sum the costs from the dictionary
                    total_dict_cost = 0
                    for v in change_value.values():
                        try:
                            total_dict_cost += float(v)
                        except Exception:
                            continue
                    cost_value = total_dict_cost
                else:
                    try:
                        cost_value = float(change_value)
                    except Exception:
                        cost_value = np.nan
                denied_results_df.loc[index, 'cost'] = cost_value
        print("\nRecourse analysis complete.")

    # Average Recourse Cost
    print("2. Average Recourse Cost (Mean cost to flip denial to approval):")
    avg_race_cost = denied_results_df.groupby('race')['cost'].mean().sort_values(ascending=False)
    print("    By Race:\n", avg_race_cost, "\n")

    print("1. Recourse Burden Gap:")
    privileged_race_cost = avg_race_cost.loc[avg_race_cost.index.intersection(privileged_races)].mean()
    unprivileged_race_cost = avg_race_cost.loc[~avg_race_cost.index.isin(privileged_races)].mean()
    race_burden_gap = unprivileged_race_cost - privileged_race_cost
    print(f"    Recourse Burden Gap by Race (Unpriv - Priv): {race_burden_gap:.4f}")

    print("\n")

# --- Helper function for Equalized Odds post-processing ---
def get_tpr_fpr(y_true, y_pred):
    """
    Calculates True Positive Rate and False Positive Rate.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return tpr, fpr

# --- Advanced Post-Processing for Equalized Odds ---
def post_process_equalized_odds(y_proba, y_true, sensitive_data, privileged_groups):
    """
    Adjusts thresholds for unprivileged groups to equalize TPR.
    """
    y_pred_fairer = (y_proba > 0.5).astype(int)

    privileged_mask = sensitive_data.isin(privileged_groups)
    privileged_true = y_true[privileged_mask].values
    privileged_pred = y_pred_fairer[privileged_mask]
    if len(privileged_true) > 0 and (privileged_true == 1).sum() > 0:
        privileged_tpr, _ = get_tpr_fpr(privileged_true, privileged_pred)
    else:
        privileged_tpr = 0

    unprivileged_groups = [g for g in sensitive_data.unique() if g not in privileged_groups]
    for group in unprivileged_groups:
        group_mask = (sensitive_data == group)
        group_true = y_true[group_mask].values
        group_proba = y_proba[group_mask]

        if len(group_true) > 0 and (group_true == 1).sum() > 0:
            thresholds = np.linspace(0.01, 0.99, 100)
            best_threshold = 0.5
            min_tpr_diff = np.inf
            for t in thresholds:
                group_pred = (group_proba > t).astype(int)
                group_tpr, _ = get_tpr_fpr(group_true, group_pred)
                tpr_diff = abs(privileged_tpr - group_tpr)
                if tpr_diff < min_tpr_diff:
                    min_tpr_diff = tpr_diff
                    best_threshold = t

            y_pred_fairer[group_mask] = (group_proba > best_threshold).astype(int)

    return y_pred_fairer

y_pred_fairer_race = post_process_equalized_odds(
    y_proba_dicr, y_test, sensitive_test['derived_race'], privileged_races
)

fairer_results_race = pd.DataFrame({
    'y_true': y_test,
    'y_pred': y_pred_fairer_race,
    'race': sensitive_test['derived_race'].values,
    'ethnicity': sensitive_test['derived_ethnicity'].values
})

fairer_results_race = fairer_results_race.dropna(subset=['race', 'ethnicity'])

denied_applicants = X_test[y_test == 0] 

print("\n--- Recalculating Metrics with the FAIRER (Recourse-Adjusted) Model by Race ---")
get_metrics_for_model(fairer_results_race, 'Fairer Model (Recourse)', denied_applicants, fairer_features, X_train)

# --- 6. Visualizations for Fairness Metrics ---
print("\n--- Generating Visualizations ---")

plt.figure(figsize=(10, 8))
feat_imp = pd.DataFrame({'Feature': X_train.columns, 'Importance': model_dicr.feature_importances_})
feat_imp = feat_imp.sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis', hue='Feature', legend=False)
plt.title('Feature Importance - Fairer LightGBM Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.close()

metrics_plot_data_race = fairer_results_race.groupby('race').apply(
    lambda x: pd.Series({'TPR': recall_score(x['y_true'], x['y_pred'], zero_division=0)}),
    include_groups=False
).reset_index().rename(columns={'race': 'Group'})
metrics_plot_data_race['Metric'] = 'TPR'

metrics_plot_data_race_fpr = fairer_results_race.groupby('race').apply(
    lambda x: pd.Series({'FPR': (x['y_pred'][x['y_true'] == 0] == 1).sum() / (x['y_true'] == 0).sum() if (x['y_true'] == 0).sum() > 0 else 0}),
    include_groups=False
).reset_index().rename(columns={'race': 'Group'})
metrics_plot_data_race_fpr['Metric'] = 'FPR'

metrics_plot_data = pd.concat([
    metrics_plot_data_race,
    metrics_plot_data_race_fpr.rename(columns={'FPR': 'TPR'})
])

recourse_results_df = fairer_results_race.loc[denied_applicants.index].copy()
recourse_results_df['cost'] = np.nan
if not denied_applicants.empty:
    for index, row in denied_applicants.iterrows():
        _, _, change_value, _ = get_minimal_counterfactual(
            model_dicr, row[fairer_features], mutable_features, X_train
        )
        if change_value is not None:
            cost_value = np.nan
            if isinstance(change_value, (pd.Series, np.ndarray, list)):
                try:
                    cost_value = float(np.ravel(change_value)[0])
                except Exception:
                    cost_value = np.nan
            elif isinstance(change_value, dict):
                total_dict_cost = 0
                for v in change_value.values():
                    try:
                        total_dict_cost += float(v)
                    except Exception:
                        continue
                cost_value = total_dict_cost
            else:
                try:
                    cost_value = float(change_value)
                except Exception:
                    cost_value = np.nan
            recourse_results_df.loc[index, 'cost'] = cost_value

print("\nAnalysis complete.")
