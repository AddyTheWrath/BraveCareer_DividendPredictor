# Plot a bar chart to visualize feature importance
'''
# plt.figure(figsize=(20, 10))
# sns.barplot(data=importance_table, x="Feature", y="Importance")
# plt.title("Feature Importance")
# plt.subplots_adjust(bottom=0.2, top=0.95)
# plt.xticks(rotation=45, ha='right', rotation_mode="anchor")
# plt.show()
'''

'''
# Now let's remove the features one by one from the least important one
X_train_temp = X_train_oversampled.copy()
X_validate_temp = X_validate_transformed.copy()

# Initialize the result dataframe
result_df = pd.DataFrame(columns=['Features_Removed', 'ROC_Score'])


# First, evaluate performance using all features
randomForestModel = RandomForestClassifier(max_features=None)
randomForestModel.fit(X_train_temp, y_train_oversampled)
# Predict probabilities on test data
y_pred_probs = randomForestModel.predict_proba(X_validate_temp)[:, 1]
# Compute ROC score
roc_score = roc_auc_score(y_test, y_pred_probs)
# Append the result to the result dataframe
result_df = pd.concat([result_df, pd.DataFrame([{'Features_Removed': 'None', 'ROC_Score': roc_score}])], ignore_index=True)
print(f"Feature_Removed: None, Number of features used: {len(X_train_temp.columns)}, ROC_AUC_Score: {roc_score}")

# Sort importance_table by Importance in ascending order to start with the least important
importance_table_sorted = importance_table.sort_values('Importance')

# Loop through features, starting from the least important
for index, row in importance_table_sorted.iterrows():
    # Drop the feature from training and test data
    X_train_temp = X_train_temp.drop(columns=[row['Feature']])
    X_validate_temp = X_validate_temp.drop(columns=[row['Feature']])
    # Train a random forest model
    randomForestModel = RandomForestClassifier(max_features=None)
    randomForestModel.fit(X_train_temp, y_train_oversampled)
    # Predict probabilities on test data
    y_pred_probs = randomForestModel.predict_proba(X_validate_temp)[:, 1]
    # Compute ROC score
    roc_score = roc_auc_score(y_test, y_pred_probs)
    # Append the result to the result dataframe
    result_df = pd.concat([result_df, pd.DataFrame([{'Features_Removed': row['Feature'], 'ROC_Score': roc_score}])],
                          ignore_index=True)
    print(
        f"Feature_Removed: {row['Feature']}, Number of features used: {len(X_train_temp.columns)}, ROC_AUC_Score: {roc_score}")
    # If only one feature left, break the loop
    if X_train_temp.shape[1] == 1:
        break

# Save the results
 with open('result_df.pkl', 'wb') as file:
    pickle.dump(result_df, file)
with open('importance_table_sorted.pkl', 'wb') as file:
    pickle.dump(importance_table_sorted, file)

# Load the results
with open('result_df.pkl', 'rb') as file:
    result_df = pickle.load(file)
'''