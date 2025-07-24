import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

"""Problem 1: Linear Regression from Scratch"""
# Implement linear regression using gradient descent without using sklearn's
#   LinearRegression

# Generate sample data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,\
                                                    random_state=42)
X_train_flat = X_train.ravel()
X_test_flat  = X_test.ravel()

# Tasks:
# 1. Implement a Linear Regression class with methods:
#       __init__(self, learning_rate=0.01, n_iterations=1000)
#       fit(self, X, y) - implements gradient descent
#       predict(self, X) - makes predictions
#       cost_history - tracks cost function over iterations
# 2. Train your model and compare results with sklearns's Linear Regression
# 3. Plot the cost function over iterations to verify convergence
# 4. Implement both bath gradient descent and stochastic gradient descent
#       versions.

# TASK 1
class ScratchLinearRegression():
    def __init__(self, learning_rate = 0.01, n_iterations=1000,\
                 method="batch",tol=1e-6):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.cost_history_values = []
        self.m_vals = []
        self.b_vals = []
        self.method = method

    def fit(self, X, y): #Implements gradient descent
        m_test = False
        b_test = False
        its = 1

        m = np.random.randint(-10,10)
        b = np.random.randint(-10,10)
        prev_m = m
        prev_b = b

        cost_vals = []

        if np.ndim(X) != 1:
            print("X must be 1D")
            return
        if np.ndim(y) != 1:
            print("y must be 1D")
            return
        
        n = len(X)

        if self.method != "batch" and self.method != "stochastic":
            print("Method must be batch or stochastic.")
            return

        while its <= self.n_iterations:
            if self.method == "batch":
                cost = 0
                for i in range(n):
                    cost += (y[i]-(m*X[i]+b))**2
                cost = (1/n)*cost
                cost_vals.append(cost)

                dcdm = 0
                dcdb = 0
                for i in range(n):
                    dcdm += X[i]*(y[i]-(m*X[i]+b))
                    dcdb += y[i]-(m*X[i]+b)

                dcdm = (-2/n)*dcdm
                dcdb = (-2/n)*dcdb

                m = m-self.learning_rate*dcdm
                b = b-self.learning_rate*dcdb

            if self.method == "stochastic":
                for i in range(n):
                    y_pred_i = m*X[i]+b
                    err = y[i] - y_pred_i

                    dcdm = -2 * X[i] * err
                    dcdb = -2 * err

                    m = m-self.learning_rate*dcdm
                    b = b-self.learning_rate*dcdb

            if its != 1:
                m_test = abs(m-prev_m) < self.tol
                b_test = abs(b-prev_b) < self.tol

            if m_test and b_test:
                self.m_vals.append(m)
                self.b_vals.append(b)
                self.cost_history_values.append(cost_vals)
            else:
                prev_m = m
                prev_b = b


            its += 1
        print("Maximum iterations reached without converging below tolerance.")
        print("Saving most recent m, b, and cost values...")
        self.m_vals.append(m)
        self.b_vals.append(b)
        self.cost_history_values.append(cost_vals)
    
    def predict(self, X): #Makes predictions
        m_avg = np.average(self.m_vals)
        b_avg = np.average(self.b_vals)
        y_predicted = m_avg*X+b_avg
        return y_predicted
    
    def cost_history(self):
        return self.cost_history_values
"""
plt.plot(X_train, y_train, 'o', label="X_train, y_train")
plt.plot(X_test, y_test, 'o', label="X_test y_test")
plt.legend()
plt.show()
"""

#Train model

LinearRegressionModel = ScratchLinearRegression(method="stochastic")

LinearRegressionModel.fit(X_train_flat, y_train)

#Get cost history
cost_history = LinearRegressionModel.cost_history()
cost_history_shape = np.shape(cost_history)

#Predict test data
y_predicted = LinearRegressionModel.predict(X_test_flat)

#Get MSE
MSE = (1/len(y_predicted))*np.sum((y_predicted-y_test)**2)

#Comparison to Scikit method
reg = LinearRegression()
reg.fit(X_train, y_train)
y_predicted_scikit = reg.predict(X_test)
MSE_Scikit = (1/len(y_predicted_scikit))*np.sum((y_predicted_scikit-y_test)**2)

MSE_diff = abs(MSE-MSE_Scikit)
print(f"The absolute difference between the Regression from scratch and the Scikit model is {MSE_diff:.3}")

#Create linear regression line
m_trained = np.mean(LinearRegressionModel.m_vals)
b_trained = np.mean(LinearRegressionModel.b_vals)
dummy_x_trained_vector = np.linspace(X.min(), X.max(),100)
dummy_y_trained_vector = m_trained*dummy_x_trained_vector + b_trained
dummy_y_scikit_trained_vector = reg.coef_[0]*dummy_x_trained_vector +\
    reg.intercept_

plt.plot(X_train_flat, y_train, 'bo', label="Training Data")
plt.plot(X_test_flat, y_test, 'g+', label="Testing Data")
plt.plot(dummy_x_trained_vector, dummy_y_trained_vector, 'r',\
         label="Regression Line")
plt.plot(dummy_x_trained_vector, dummy_y_scikit_trained_vector, 'k--',\
         label="Scikit Line")
plt.legend()
plt.show()

"""
#Plot cost for convergence
for i in range(cost_history_shape[0]):
    plt.plot(cost_history[i][:], 'o', label="Cost over iterations")
plt.show()
"""