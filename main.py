import pandas as pd
import numpy as np
import os
import warnings

# import matplotlib.pyplot as plt
# import seaborn as sns

from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score # make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, GridSearchCV
import pickle
import optuna
from xgboost import XGBClassifier

# from company_data_extractor import company_data_extractor


# Register API for Financial Modeling Prep (Financial Statements and Company Fundamentals)
# https://site.financialmodelingprep.com/developer/
# Register API for Federal Reserve Economic Data (For Macroeconomics Data)
# https://fred.stlouisfed.org/docs/api/fred/
# Yahoo Finance does not need an API

warnings.filterwarnings('ignore')


# -------------------------------------------------- Start from here --------------------------------------------------

# Load Data
dataset = pd.read_csv("Stock_data.csv")

# Null value analysis
print("1. Loaded dataset information:")
dataset.info(verbose=True)

# Multivariate Analysis

# Selecting columns where float or integer
numeric_dataset = dataset.select_dtypes(include=[np.number])

# Creating orrelation matrix
correlation_matrix = numeric_dataset.corr()

def rank_columns_by_correlation(df, threshold=0.9):
    # Calculating correlation matrix
    corr_matrix = df.corr()
    # Initializing a list to hold the tuples (col1, col2, correlation)
    correlations = []
    # Iterating over the correlation matrix
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):  # avoiding duplicate and self-correlation
            # Including only correlations above the specified threshold
            if abs(corr_matrix.iloc[i, j]) > threshold:
                correlations.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

    # Sorting the list by absolute correlation in descending order
    sorted_correlations = sorted(correlations, key=lambda x: abs(x[2]), reverse=True)
    correlation_df = pd.DataFrame(sorted_correlations, columns=['Column1', 'Column2', 'Correlation'])
    return correlation_df

top_correlations = rank_columns_by_correlation(numeric_dataset, 0.98)

# Remove highly correlated columns
columns_to_remove = top_correlations["Column2"].unique()
dataset.drop(columns_to_remove, axis="columns", inplace=True)

# Data Preprocessing
# Missing values
print("2. High correlations have been removed. Dataset information:")
dataset.info(verbose=True)

# Drop NA
dataset.dropna(inplace=True)

# First let's leave out the last year's data as future test data, and 2021's data as validation data
training_data = dataset.loc[(dataset["year"] != 2022) & (dataset["year"] != 2021)]
validation_data = dataset.loc[dataset["year"] == 2021]
testing_data = dataset.loc[dataset["year"] == 2022]


# Predictor - Target Split
X_train = training_data.drop("dps_trend", axis="columns")
y_train = training_data["dps_trend"]
X_test = testing_data.drop("dps_trend", axis="columns")
y_test = testing_data["dps_trend"]
X_validate = validation_data.drop("dps_trend", axis="columns")
y_validate = validation_data["dps_trend"]

# Encoding our categorical features - not all models support them

# Define categorical features
categorical_columns = ["industry", "sector", "symbol"]
other_columns = [col for col in X_train.columns if col not in categorical_columns]

# Label encode categorical features with many categories
column_transformer = ColumnTransformer(
    transformers=[
        ('categorical', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_columns)
    ],
    remainder='passthrough'
)

X_train_transformed = column_transformer.fit_transform(X_train)
X_validate_transformed = column_transformer.transform(X_validate)
X_test_transformed = column_transformer.transform(X_test)

# Note: after transformation, the output will be a numpy array and column orders will be changed.
X_train_transformed = pd.DataFrame(X_train_transformed, columns=categorical_columns + other_columns)
X_validate_transformed = pd.DataFrame(X_validate_transformed, columns=categorical_columns + other_columns)
X_test_transformed = pd.DataFrame(X_test_transformed, columns=categorical_columns + other_columns)

# Check our data type
print("3. Transformed dataset (categorical to ordinal):")

print(X_train_transformed.head())

# Let's change our data types back to their original forms - However, this time, categorical variables have become
# number-like strings
cols_to_convert = {'industry': 'str', 'sector': 'str', 'symbol': 'str', 'year': 'int'}
X_train_transformed = X_train_transformed.astype(cols_to_convert)
X_validate_transformed = X_validate_transformed.astype(cols_to_convert)
X_test_transformed = X_test_transformed.astype(cols_to_convert)

# Check data imbalance
# Let's add target back to our dataset for further analysis
training_data_transformed = pd.concat([X_train_transformed, y_train], axis=1)

print("4. Value counts for cleaned, transformed data (source and target):")
print(training_data_transformed["dps_trend"].value_counts())

# Let's do some over sampling

# Perform oversampling using SMOTE

# Identifying indices (column #s) where categorical
categorical_indices = [X_train_transformed.columns.get_loc(col) for col in categorical_columns]

# Creating a smote object by feeding categorcal indices
smote = SMOTENC(random_state=1, categorical_features=categorical_indices)

# Retrieiving x train and y train oversampled by fitting and resampling the transformed data to the SMOTE object
X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train_transformed, y_train)

# Check our training data
print("5. Oversampled y-train value counts:")
print(pd.DataFrame(y_train_oversampled)["dps_trend"].value_counts())

print("6. Oversampled x-train information:")
X_train_oversampled.info()

# Feature selection

# Feature importance analysis - Tree Based
randomForestModel = RandomForestClassifier(max_features=None)  # We want all features to be considered for each tree

# Fitting the oversampled data to the RF model
randomForestModel.fit(X_train_oversampled, y_train_oversampled)

# Retrieve NumPy array which stores a "score" for each feature in the training set
model_importance = randomForestModel.feature_importances_

# Create an importance table
importance_table = pd.DataFrame(columns=["Feature", "Importance"])
featureNum = 0
for score in model_importance:
    print("feature " + str(featureNum) + "'s importance score: " + str(score) + " (" + X_train_oversampled.columns[featureNum] + ")")
    rowAdded = pd.DataFrame([[X_train_oversampled.columns[featureNum], score]], columns=["Feature", "Importance"])
    importance_table = pd.concat([importance_table, rowAdded])
    featureNum = featureNum + 1

# Sort the table by importance score
importance_table.sort_values('Importance', inplace=True, ascending=True)

print(importance_table.head)

# --- CODE SNIPPET CUT OUT - SEE NOTES FOR REFERENCE - 13 MOST IMPORTANT FEATURES RESULT IN BEST ROC AUC SCORE ---

# Get the first 87 features
least_important_features = importance_table['Feature'].iloc[:87]

# Drop them for KNN
X_train_reduced = X_train_oversampled.drop(columns=least_important_features)
X_validate_reduced = X_validate_transformed.drop(columns=least_important_features)
X_test_reduced = X_test_transformed.drop(columns=least_important_features)

# Model Selection

# Scale the features
scaler = StandardScaler() # For LR
scaler2 = StandardScaler() # For KNN

X_train_scaled = scaler.fit_transform(X_train_oversampled) # For LR
X_train_reduced_scaled = scaler2.fit_transform(X_train_reduced) # For KNN

X_validate_scaled = scaler.transform(X_validate_transformed) # For LR
X_validate_reduced_scaled = scaler2.transform(X_validate_reduced) # For KNN

X_test_scaled = scaler.transform(X_test_transformed) # For LR
X_test_reduced_scaled = scaler2.transform(X_test_reduced) # For KNN

# GridSearch - not being used.
'''
param_grid = {
    "penalty": ['l1', 'l2'],  # These have to be the same as the estimator's parameters' name
    "C": np.arange(0.1, 10, 0.1).tolist()
}
gridSearch = GridSearchCV(estimator=LogisticRegression(random_state=1), param_grid=param_grid, scoring='roc_auc',
                          cv=5, n_jobs=-1)
gridSearch.fit(X_train_scaled, y_train_oversampled)

best_params_lr = gridSearch.best_params_
print("Best Logistic Regression Parameters using Grid Search: ", best_params_lr)
print("Best Logistic Regression ROC-AUC Score using Grid Search: ", gridSearch.best_score_)

# Create model with best parameters found during grid search
best_model_lr = LogisticRegression(**best_params_lr, solver='liblinear', n_jobs=-1)

# Save model using pickle. First open a file called lr.pkl in write/binary mode, and then dump the best_model_lr into it.
with open('best_models/lr.pkl', 'wb') as file:
    pickle.dump(best_model_lr, file)
'''

# Bayesian Optimization with optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress log messages that are not warnings

# Logistic Regression
def objective_function(trial):
    c = trial.suggest_float('C', 0.1, 10, log=True)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])

    model = LogisticRegression(
        C=c,
        penalty=penalty,
        solver='liblinear',
        n_jobs=-1
    )

    # Using cross_val_score to get the average precision score for each fold
    scores = cross_val_score(model, X_train_scaled, y_train_oversampled, cv=5, scoring='roc_auc')
    roc_auc = np.mean(scores)
    # Printing intermediate results
    print(f"Trial {trial.number}, C: {c}, penalty: {penalty}, ROC-AUC: {roc_auc}")
    return roc_auc

study_lr = optuna.create_study(direction="maximize")
study_lr.optimize(objective_function, n_trials=100)

best_params_lr = study_lr.best_params
print("Best Logistic Regression Parameters using Bayesian Optimization: ", best_params_lr)
print("Best Logistic Regression ROC-AUC Score using Bayesian Optimization: ", study_lr.best_value)

# Create and save model
os.makedirs('best_models', exist_ok=True)
best_model_lr = LogisticRegression(**best_params_lr, solver='liblinear', n_jobs=-1)
with open('best_models/lr.pkl', 'wb') as file:
    pickle.dump(best_model_lr, file)

print ("Logistic Regression model saved.")

# Decision Tree
def objective_function(trial):
    max_depth = trial.suggest_int('max_depth', 1, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 15)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 15)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion
    )

    # Using cross_val_score to get the average precision score for each fold
    scores = cross_val_score(model, X_train_oversampled, y_train_oversampled, cv=5, scoring='roc_auc')
    roc_auc = np.mean(scores)
    # Printing intermediate results
    print(f"Trial {trial.number}, max_depth: {max_depth}, min_samples_split: {min_samples_split}, "
          f"min_samples_leaf: {min_samples_leaf}, criterion: {criterion}, ROC-AUC: {roc_auc}")
    return roc_auc

study_dt = optuna.create_study(direction="maximize")
study_dt.optimize(objective_function, n_trials=100)

best_params_dt = study_dt.best_params
print("Best Parameters: ", best_params_dt)
print("Best ROC-AUC Score: ", study_dt.best_value)

# Create and save model
best_model_dt = DecisionTreeClassifier(**best_params_dt)
with open('best_models/dt.pkl', 'wb') as file:
    pickle.dump(best_model_dt, file)

# KNN
def objective_function(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 1, 30)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    p = trial.suggest_int('p', 1, 5)
    metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        p=p,
        metric=metric
    )

    # Using cross_val_score to get the average precision score for each fold
    scores = cross_val_score(model, X_train_reduced_scaled, y_train_oversampled, cv=5, scoring='roc_auc')
    roc_auc = np.mean(scores)
    # Printing intermediate results
    print(f"Trial {trial.number}, n_neighbors: {n_neighbors}, weights: {weights}, p: {p}, metric: {metric}, "
          f"ROC-AUC: {roc_auc}")
    return roc_auc

study_knn = optuna.create_study(direction="maximize")
study_knn.optimize(objective_function, n_trials=100)

best_params_knn = study_knn.best_params
print("Best Parameters: ", best_params_knn)
print("Best ROC-AUC Score: ", study_knn.best_value)

# Create and save model
best_model_knn = KNeighborsClassifier(**best_params_knn)
with open('best_models/knn.pkl', 'wb') as file:
    pickle.dump(best_model_knn, file)

# Random Forest
def objective_function(trial):
    n_estimators = trial.suggest_int('n_estimators', 2, 150)
    max_depth = trial.suggest_int('max_depth', 1, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 15)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 15)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        n_jobs=-1
    )

    # Using cross_val_score to get the average ROC-AUC score for each fold
    scores = cross_val_score(model, X_train_oversampled, y_train_oversampled, cv=5, scoring='roc_auc')
    roc_auc = np.mean(scores)
    # Printing intermediate results
    print(f"Trial {trial.number}, n_estimators: {n_estimators}, max_depth: {max_depth}, "
          f"min_samples_split: {min_samples_split}, min_samples_leaf: {min_samples_leaf}, criterion: {criterion}, ROC-AUC: {roc_auc}")
    return roc_auc

study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(objective_function, n_trials=100)

best_params_rf = study_rf.best_params
print("Best Parameters: ", best_params_rf)
print("Best ROC-AUC: Score: ", study_rf.best_value)

# Create and save model
best_model_rf = RandomForestClassifier(**best_params_rf, n_jobs=-1)
with open('best_models/rf.pkl', 'wb') as file:
    pickle.dump(best_model_rf, file)

# XgBoost
# It requires the target to be 0 and 1, and all features be numerical
# Encode our target
label_encoder = LabelEncoder()
# Fit the encoder and transform the target variable
y_train_oversampled_encoded = label_encoder.fit_transform(y_train_oversampled)
y_validate_encoded = label_encoder.transform(y_validate)
y_test_encoded = label_encoder.transform(y_test)

# Cast categorical types into numbers

cols_to_convert = {'sector': 'float', 'industry': 'float', 'symbol': 'float'}
X_train_xg_boost = X_train_oversampled.astype(cols_to_convert)
X_validate_xg_boost = X_validate_transformed.astype(cols_to_convert)
X_test_xg_boost = X_test_transformed.astype(cols_to_convert)

# Suppress printing logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective_function(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 1000)
    max_depth = trial.suggest_int('max_depth', 1, 50)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.9, log=True)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    gamma = trial.suggest_float('gamma', 0, 1.0)
    reg_alpha = trial.suggest_float('reg_alpha', 0, 1)
    reg_lambda = trial.suggest_float('reg_lambda', 0, 1)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        use_label_encoder=False,
        n_jobs=-1
    )

    # Using cross_val_score to get the average ROC-AUC score for each fold
    scores = cross_val_score(model, X_train_xg_boost, y_train_oversampled_encoded, cv=5, scoring='roc_auc')
    roc_auc = np.mean(scores)
    # Printing intermediate results
    print(f"Trial {trial.number}, n_estimators: {n_estimators}, max_depth: {max_depth}, learning_rate: {learning_rate}," 
          f"min_child_weight: {min_child_weight}, subsample: {subsample}, colsample_bytree: {colsample_bytree}, "
          f"gamma: {gamma}, reg_alpha: {reg_alpha}, reg_lambda: {reg_lambda}, ROC-AUC: {roc_auc}")
    return roc_auc

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_function, n_trials=100)

best_params_xgb = study_xgb.best_params
print("Best Parameters: ", best_params_xgb)
print("Best ROC-AUC Score: ", study_xgb.best_value)

best_model_xgb = XGBClassifier(**best_params_xgb, use_label_encoder=False, n_jobs=-1)
with open('best_models/xgb.pkl', 'wb') as file:
    pickle.dump(best_model_xgb, file)

# Model selection - Compare Performance
with open('best_models/lr.pkl', 'rb') as file:
    best_model_lr = pickle.load(file)
with open('best_models/dt.pkl', 'rb') as file:
    best_model_dt = pickle.load(file)
with open('best_models/knn.pkl', 'rb') as file:
    best_model_knn = pickle.load(file)
with open('best_models/rf.pkl', 'rb') as file:
    best_model_rf = pickle.load(file)
with open('best_models/xgb.pkl', 'rb') as file:
    best_model_xgb = pickle.load(file)

print("Testing Performances on Validation Set...Please wait")
best_model_lr.fit(X_train_scaled, y_train_oversampled)
predicted_probs = best_model_lr.predict_proba(X_validate_scaled)[:, 1]
lr_performance = roc_auc_score(y_validate, predicted_probs)

best_model_dt.fit(X_train_oversampled, y_train_oversampled)
predicted_probs = best_model_dt.predict_proba(X_validate_transformed)[:, 1]
dt_performance = roc_auc_score(y_validate, predicted_probs)

best_model_knn.fit(X_train_reduced_scaled, y_train_oversampled)
predicted_probs = best_model_knn.predict_proba(X_validate_reduced_scaled)[:, 1]
knn_performance = roc_auc_score(y_validate, predicted_probs)

best_model_rf.fit(X_train_oversampled, y_train_oversampled)
predicted_probs = best_model_rf.predict_proba(X_validate_transformed)[:, 1]
rf_performance = roc_auc_score(y_validate, predicted_probs)

best_model_xgb.fit(X_train_xg_boost, y_train_oversampled_encoded)
predicted_probs = best_model_xgb.predict_proba(X_validate_xg_boost)[:, 1]
xgb_performance = roc_auc_score(y_validate_encoded, predicted_probs)

# Performance of the models on the validation set are
print(f"Logistic Regression Test ROCAUC: {lr_performance}")
print(f"Decision Tree Test ROCAUC: {dt_performance}")
print(f"KNN Test ROCAUC: {knn_performance}")
print(f"Random Forest Test ROCAUC: {rf_performance}")
print(f"XGBoost Test ROCAUC: {xgb_performance}")

# Build the final pipeline for production
pipeline = Pipeline(steps=[
                           ('classifier', best_model_xgb)
                          ])

result = pipeline.predict(X_test_xg_boost)

y_pred_proba = pipeline.predict_proba(X_test_xg_boost)[:, 1]

roc_auc = roc_auc_score(y_test_encoded, y_pred_proba)
print(f"Final ROC AUC Score on Test Data: {roc_auc}")