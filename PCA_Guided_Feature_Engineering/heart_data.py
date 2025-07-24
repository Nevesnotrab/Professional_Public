from PCAGuidedFeatureBuilder import *
from sklearn.preprocessing import StandardScaler
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Get data
heart_data = pd.read_csv('heart.csv')
target_col = 'target'

# Define X and y
X = heart_data.copy()
X.drop([target_col], axis=1, inplace=True)

categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

y = heart_data[target_col]
print(y.mean())
print(y.std())

# Scale features
scaler = StandardScaler()
X_numeric_scaled = pd.DataFrame(scaler.fit_transform(X[numeric_cols]), columns=numeric_cols, index=X.index)
X_scaled = pd.concat([X_numeric_scaled, X[categorical_cols]], axis=1)

# Set up cross-validation
random_state = 42
kf = KFold(n_splits = 5, shuffle=True, random_state=random_state)

# Arrays to collect errors
baseline_acc_list = []
engineered_acc_list = []

# PCAGuidedFeatureBuilder method
method='mean'

for train_idx, test_idx in kf.split(X_scaled):
    X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Baseline model fit, predict, acc
    baseline_model = XGBClassifier()
    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)
    baseline_acc = accuracy_score(y_test, y_pred_baseline)
    baseline_acc_list.append(baseline_acc)

    # Get PCA-informed features from PCAGuidedFeatureBuilder
    builder = PCAGuidedFeatureBuilderClass(X=X_train,method=method)
    engineered_features_train = builder.main()
    X_PCA_train = pd.concat([X_train, engineered_features_train],axis=1)

    # Apply transformation to X_test
    engineered_features_test = builder.Transform(X_test)
    X_PCA_test = pd.concat([X_test, engineered_features_test],axis=1)

    # Fit, predict, acc
    engineered_model = XGBClassifier()
    engineered_model.fit(X_PCA_train,y_train)
    y_pred_engineered = engineered_model.predict(X_PCA_test)
    engineered_acc = accuracy_score(y_test,y_pred_engineered)
    engineered_acc_list.append(engineered_acc)
    
print("Mean baseline acc: ", np.mean(baseline_acc_list))
print("Mean Engineered acc: ", np.mean(engineered_acc_list))
improvement_percent = 100 * (np.mean(baseline_acc_list)-np.mean(engineered_acc_list))/np.mean(baseline_acc_list)
print(f"Relative acc improvement (%): {improvement_percent:2f}")