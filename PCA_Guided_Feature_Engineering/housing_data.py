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
from category_encoders import MEstimateEncoder

# Get data
house_data = pd.read_csv('housing.csv')

# Define X and y
target_col_name = 'median_house_value'
cat_cols = ['ocean_proximity']
X_master = house_data.copy()
X_master['total_bedrooms'].fillna(X_master['total_bedrooms'].median(),inplace=True)
y_master = X_master.pop(target_col_name)
numeric_cols = [col for col in X_master.columns if col not in cat_cols]

random_state = 42
kf = KFold(n_splits = 20, shuffle=True, random_state=random_state)

#Arrays to collect errors
baseline_mae_list = []
engineered_mae_list = []

# PCAGuidedFeatureBuilderMethod
method = 'mean'

for train_idx, test_idx in kf.split(X_master):
    X_pretrain, X_pretest = X_master.iloc[train_idx], X_master.iloc[test_idx]
    y_train, y_test = y_master.iloc[train_idx], y_master.iloc[test_idx]

    X_encode = X_pretrain.sample(frac=0.5,random_state=random_state)
    y_encode = y_train.loc[X_encode.index]

    X_pretrain_2 = X_pretrain.drop(X_encode.index)
    y_train_2 = y_train.drop(X_encode.index)

    encoder = MEstimateEncoder(cols=cat_cols,m=5.0)
    encoder.fit(X_encode, y_encode)

    X_train_final = encoder.transform(X_pretrain_2)
    X_test_final  = encoder.transform(X_pretest)

    x_scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        x_scaler.fit_transform(X_train_final),
        columns=X_train_final.columns,
        index=X_train_final.index
    )

    X_test_scaled = pd.DataFrame(
        x_scaler.transform(X_test_final),
        columns=X_test_final.columns,
        index=X_test_final.index
    )

    # Baseline model fit, predict, MAE
    baseline_model = XGBRegressor()
    baseline_model.fit(X_train_scaled,y_train_2)
    y_pred_baseline = baseline_model.predict(X_test_scaled)
    baseline_mae = mean_absolute_error(y_test,y_pred_baseline)
    baseline_mae_list.append(baseline_mae)

    
    # Get PCA-informed features from PCAGuidedFeatureBuilder
    builder = PCAGuidedFeatureBuilderClass(X=X_train_scaled,method=method)
    engineered_features_train = builder.main()
    X_PCA_train = pd.concat([X_train_scaled,engineered_features_train],axis=1)

    # Apply transformation to X_test
    engineered_features_test = builder.Transform(X_test_scaled)
    X_PCA_test = pd.concat([X_test_scaled, engineered_features_test], axis=1)

    # Fit, predict, etc.
    engineered_model = XGBRegressor()
    engineered_model.fit(X_PCA_train,y_train_2)
    y_pred_engineered = engineered_model.predict(X_PCA_test)
    engineered_mae = mean_absolute_error(y_test,y_pred_engineered)
    engineered_mae_list.append(engineered_mae)
    
print("Mean baseline MAE: ", np.mean(baseline_mae))
print("Stdev baseline MAE: ", np.std(baseline_mae_list))
print("Mean Engineered MAE: ", np.mean(engineered_mae_list))
print("Stdev Engineered MAE: ", np.std(engineered_mae_list))
improvement_percent = 100 * (np.mean(baseline_mae)-np.mean(engineered_mae_list))/np.mean(baseline_mae)
print(f"Relative MAE improvement: (%): {improvement_percent:2f}")