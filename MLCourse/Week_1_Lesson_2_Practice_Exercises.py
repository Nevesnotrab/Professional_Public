"""Exercise 1 - Run the code in Week 1 Lesson 2 Hands On and experiment with different functions."""
#Done

"""Exercise 2 - Manual Gradient Calculation."""
"""For f(x,y) = 2x^2 + 2xy + y^2, calculate:
    a. ∂f/∂x by hand
    b. ∂f/∂y by hand
    c. The gradient vector ∇f
    d. Verify with the numerical gradient function"""

# a. df/dx = 4x+2y
# b. df/dy = 2y+2x
# c. The gradient vector will be [df/dx, df/dy]

import numpy as np
import matplotlib.pyplot as plt

def DerivativeApproximation(f, x, h=0.001):
    gradvector = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        gradvector[i] = (f(x_plus) - f(x_minus))/(2*h)
    return gradvector

def f(x):
    return 2*x[0]**2 + 2*x[0]*x[1] + x[1]**2

x0 = np.linspace(-10, 10, 101)
x1 = np.linspace(-10, 10, 101)
x_test = np.array([x0,x1]).T

f_test_arr = np.zeros_like(x_test)
gradient_xtest = np.zeros_like(x_test)
for i in range(len(x0)):
    f_test_arr[i] = f(x_test[i,:])
    gradient_xtest[i] = DerivativeApproximation(f, x_test[i,:])

plt.plot(x0, f_test_arr[:,0])
plt.plot(x0, gradient_xtest[:,0])
plt.show()
plt.plot(x1, f_test_arr[:,1])
plt.plot(x1, gradient_xtest[:,1])
plt.show()

"""Exercise 3 - Chain Rule Practice"""
"""Given the composition f(g(x) where:
    * g(x) = x^2 + 1
    * f(u) = sin(u)
    Calculate the derivative of the composition using the chain rule."""
import math

# dg/dx = 2x
# df/du = cos(u)
# If u = g(x), then we substitute:
#   df/du * dg/dx = df/dx = 2x * cos(u) where u = g(x) = x**2 + 1.
#   df/du = 2x*cos(x**2+1)

def g(x):
    return x**2 + 1
def f(u):
    return math.sin(u)

def ChainRule(f, g, x):
    u = g(x)
    dgdx = DerivativeApproximation(g,x)[0]
    dfdu = DerivativeApproximation(f,u)[0]
    dfdx = dfdu * dgdx
    return dfdx



"""Exercise 4 - Optimization Experiment"""
"""Optimize various functions with different learning rates. Observe:
    * How Learning Rates affect convergence speed
    * What happens with learning rates that are too large
    * The difference between convex and non-convex optimization"""

# Skipped experimenting with the code because I am familiar with this
# High learning rates can improve convergence speed (iterations and runtime),
#   however numeric instability is a serious concern, as too high of a learning
#   rate can cause perpetual overshooting of the solution, resulting in poor
#   optimization results.
# Low learning rates generally slow down convergence speed, but generally avoid
#   the issue of overshooting the solution.
# Learning rate is generally a tradeoff between convergence speed and convergence
#   "reliability"
# Convex functions tend to be nice to minimize because gradient-descent methods
#   identify the minimum relatively quickly and with good accuracy.
# Non-convex functions run into gradient descent issues because they may have more
#   than one minima or maxima. This means that some points cause solutions
#   to become "stuck" at a local minima or maxima (if the starting point is a maxima).
# Non-convex functions are better-optimized by non-gradient methods or by varying
#   starting points across the domain as much as possible.