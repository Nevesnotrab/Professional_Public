"""Problem 3"""
"""k-NN with Custom Distance Metrics"""

# Implement k-NN classifier with multiple distance metrics and compare
#   performance.

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

wine = load_wine()
X, y = wine.data, wine.target

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sklearn_knn = KNeighborsClassifier(n_neighbors=3)
sklearn_knn.fit(X_train_scaled, y_train)
y_pred_sklearn = sklearn_knn.predict(X_test_scaled)
sk_accuracy = accuracy_score(y_test, y_pred_sklearn)
sklearn_unscaled_knn = KNeighborsClassifier(n_neighbors=3)
sklearn_knn.fit(X_train, y_train)
y_pred_sklearn_unscaled = sklearn_knn.predict(X_test)

# Tasks:
# 1. Implement a KNNClassifier class with:
#   * Support for Euclidean, Manhattan, and Minkowski distances
#   * fit(self, X, y) method
#   * predict(self, X) method
#   * predict_proba(self, X) method
# 2. Implement these distance functions:
#   * euclidean_distance(x1, x2)
#   * manhattan_distance(x1, x2)
#   * minkowski_distance(x1, x2, p)
# 3. Compare your implementation with sklearn's KNeighborsClassifier on the wine
#   dataset.
# 4. Analyze how different distance metrics affect performance with and without
#   feature scaling:
#   * Which distance metric performs best on this dataset
#   * How does the curse of dimensionality affect performance as you increase
#       it?
 
class KNNClassifier():
    def __init__(self, k=3):
        self.k = k
        self.X = None
        self.y = None
    
    @staticmethod
    def _specific_distance(x1, x2, p):
        if not isinstance(x1, np.ndarray) or not isinstance(x2, np.ndarray):
            raise TypeError("x1 and x2 must be NumPy arrays.")
        if x1.ndim != 1 or x2.ndim != 1:
            raise ValueError("x1 and x2 must both be 1D arrays.")
        if not np.isscalar(p):
            raise TypeError("p must be a scalar value.")
        if p==0:
            raise ValueError("p cannot be 0.")
        return np.sum(np.abs(x1-x2)**p)**(1/p)
        
    def fit(self, X, y):  
        if not (isinstance(X, np.ndarray) and isinstance(y, np.ndarray)):
            raise TypeError("X and y must be NumPy arrays.")
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be a 2D array and y must be a 1D array.")
        if X.shape[0] < self.k:
            raise ValueError("Number of samples in X must be >= k.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match.")

        self.X = X
        self.y = y
        return self
    
    def predict(self, X, method="Euclidean"):
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array.")
        if X.ndim != 1:
            raise ValueError("For predictions, X must be a 1D array.")
        
        if method not in {"Euclidean", "Manhattan", "Minkowski"}:
            raise ValueError("Method must be Euclidean, Manhattan, or Minkowski.")
        
        nearest_neighbor_indices = np.zeros(self.k, dtype=int)
        nearest_neighbor_values = np.full(self.k, np.inf)

        for i in range(self.X.shape[0]):
            if method == "Euclidean":
                dist = self._specific_distance(self.X[i,:],X,p=2)
            elif method == "Manhattan":
                dist = self._specific_distance(self.X[i,:],X,p=1)
            elif method == "Minkowski":
                dist = self._specific_distance(self.X[i,:],X,p=3)
            
            max_index = np.argmax(nearest_neighbor_values)
            if dist < nearest_neighbor_values[max_index]:
                nearest_neighbor_indices[max_index] = i
                nearest_neighbor_values[max_index] = dist
        
        nearest_neighbor_results = self.y[nearest_neighbor_indices]
        unique_values, counts = np.unique(nearest_neighbor_results, return_counts=True)
        most_probable_result = nearest_neighbor_results[np.argmax(counts)]
        return most_probable_result

CustomKNN = KNNClassifier(k=3)
CustomKNN.fit(X_train_scaled, y_train)
y_pred_custom_knn = np.array([])
y_pred_custom_knn_manhattan = np.array([])
y_pred_custom_knn_minkowski = np.array([])

for i in range(len(X_test_scaled)):
    y_pred_custom_knn = np.append(y_pred_custom_knn, CustomKNN.predict(X_test_scaled[i,:]))
    y_pred_custom_knn_manhattan = np.append(y_pred_custom_knn_manhattan, CustomKNN.predict(X_test_scaled[i,:],method="Manhattan"))
    y_pred_custom_knn_minkowski = np.append(y_pred_custom_knn_minkowski, CustomKNN.predict(X_test_scaled[i,:],method="Minkowski"))

accuracy_score_custom = accuracy_score(y_test, y_pred_custom_knn)

print(sk_accuracy)
print(accuracy_score_custom)
accuracy_score_manhattan = accuracy_score(y_test, y_pred_custom_knn_manhattan)
accuracy_score_minkowski = accuracy_score(y_test, y_pred_custom_knn_minkowski)

print("Scaled; distance method comparison:")
print("SKLearn: ", sk_accuracy)
print("Euclidean: ", accuracy_score_custom)
print("Manhattan: ", accuracy_score_manhattan)
print("Minkowski: ", accuracy_score_minkowski)

accuracy_score_unscaled_sklearn = accuracy_score(y_test, y_pred_sklearn_unscaled)
CustomKNN_Unscaled = KNNClassifier(k=3)
CustomKNN_Unscaled.fit(X_train, y_train)
y_pred_custom_knn_unscaled_euclidean = np.array([])
y_pred_custom_knn_unscaled_manhattan = np.array([])
y_pred_custom_knn_unscaled_minkowski = np.array([])

for i in range(len(X_test)):
    y_pred_custom_knn_unscaled_euclidean = np.append(y_pred_custom_knn_unscaled_euclidean, CustomKNN_Unscaled.predict(X_test[i,:]))
    y_pred_custom_knn_unscaled_manhattan = np.append(y_pred_custom_knn_unscaled_manhattan, CustomKNN_Unscaled.predict(X_test[i,:],method="Manhattan"))
    y_pred_custom_knn_unscaled_minkowski = np.append(y_pred_custom_knn_unscaled_minkowski, CustomKNN_Unscaled.predict(X_test[i,:],method="Minkowski"))

accuracy_score_unscaled_euclidean = accuracy_score(y_test, y_pred_custom_knn_unscaled_euclidean)
accuracy_score_unscaled_manhattan = accuracy_score(y_test, y_pred_custom_knn_unscaled_manhattan)
accuracy_score_unscaled_minkowski = accuracy_score(y_test, y_pred_custom_knn_unscaled_minkowski)

print("Unscaled; distance method comparison:")
print("SKLearn: ", accuracy_score_unscaled_sklearn)
print("Euclidean: ", accuracy_score_unscaled_euclidean)
print("Manhattan: ", accuracy_score_unscaled_manhattan)
print("Minkowski: ", accuracy_score_unscaled_minkowski)