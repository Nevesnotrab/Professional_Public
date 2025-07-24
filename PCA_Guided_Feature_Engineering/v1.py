from PCAGuidedFeatureBuilder import *
from sklearn.preprocessing import StandardScaler
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np

# Get data
heart_data = pd.read_csv('heart.csv')

# Define X and y
X = heart_data.copy()
X.drop(['target'], axis=1, inplace = True)
y = heart_data['target']

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Random state tests
MAEbarr = np.array([])
MAEafarr = np.array([])

for i in range(1):
    print("Run ", i)
    random_state = i
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)
    baseline_model = XGBRegressor()
    baseline_model.fit(X_train,y_train)
    pred = baseline_model.predict(X_test)
    mae = mean_absolute_error(y_test,pred)
    #print("Baseline MAE: ", mae)

    # Build PCAGuidedFeatureBuilder
    builder = PCAGuidedFeatureBuilderClass(X=X,
                                        method='mean',
                                        model=baseline_model,
                                        y=y,
                                        test_size=test_size,
                                        scoring_method=mean_absolute_error,
                                        random_state=random_state)
    EF, MAEb, MAEaf = builder.main()
    MAEbarr = np.append(MAEbarr, MAEb)
    MAEafarr = np.append(MAEafarr, MAEaf)

base_mean = np.mean(MAEbarr)
fe_mean = np.mean(MAEafarr)
print("Average baseline: ", base_mean)
print("Standard deviation baseline: ", np.std(MAEbarr))
print("Average all PCA features: ", fe_mean)
print("Standard deviation all PCA features: ", np.std(MAEafarr))
abs_dif = base_mean-fe_mean
print(abs_dif)
improvement_percent = 100 * (base_mean-fe_mean)/base_mean
print(f"Relative MAE improvement (%): {improvement_percent:2f}")

"""
input_shape = [X_train.shape[1]]
model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=input_shape),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

early_stopping = EarlyStopping(
    min_delta = 0.0001,
    patience = 20,
    restore_best_weights = True
)

history = model.fit(
    X_train, y_train,
    batch_size = 512,
    epochs=500,
    callbacks=[early_stopping],
    verbose=0
)

y_pred_deep = model.predict(X_test)
y_pred_labels = (y_pred_deep>0.5).astype(int)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred_labels)
print("Accuracy: ", accuracy)
"""