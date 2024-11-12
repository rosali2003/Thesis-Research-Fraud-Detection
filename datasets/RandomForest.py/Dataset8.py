from sklearn.ensemble import RandomForestClassifier

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Random Forest model
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Calculate Precision-Recall AUC for Random Forest
precision, recall, _ = precision_recall_curve(y_test, y_pred_rf)
pr_auc_rf = auc(recall, precision)
print(f"Random Forest Precision-Recall AUC: {pr_auc_rf:.4f}")