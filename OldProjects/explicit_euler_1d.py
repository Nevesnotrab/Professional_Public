import matplotlib.pyplot as plt
from numpy import linspace

"""
Performs an Explicit Euler Analysis on a 1-D differentiable equation.
This example assumes the original function is known just to be able to
    plot the functions at the end, but this is not necessary).
"""
#The original function (if available. x^2 is used for this example)
def f(x):
    return x * x

#The differential equation. 1-D only
def df_dx(x):
    return 2. * x
    
def EE_method(func, y0, dx, x0, num_step, min_x = 1e-6):
    #func is the function for the derivative
        #If you want a numerical derivative then read below
    #y0 is the initial f value
    #dx is the x step
    #x0 is the initial x value
    #num_steps is the number of steps to be taken
    #min_x is for calculating the numerical derivative. Don't mess with it
        #unless you know what you are doing

    #Creates blank x and f arrays
    x_new = []
    f_new = []

    #Appends the first value of x and f to their arrays
    x_new.append(x0)
    f_new.append(y0)
    
    #Performs the explicit euler
    #It starts at 1 because the first values in x and f are already done
    for i in range(1, num_step):
        #If you use this next line you need to pass func as f, not df_dx
        #It calculates a numerical derivative instead of using the known df_dx
        #dfdx = (func(x_new[i - 1] + min_x) - func(x_new[i - 1])) / min_x
        dfdx = func(x_new[i-1])
        
        #Appends the next x and f(x) values to their respective arrays
        x_new.append(x_new[i - 1] + dx)
        f_new.append(dfdx * dx + f_new[i - 1])
    
    #Returns the x and f arrays
    return x_new, f_new

#Establishes some initial conditions, step number, and dx value
x0 = 0.
ns = 100
dx = 0.1
y0 = 0.

#Runs the Explicit Euler code with the initial conditions, step number, and dx
    #values from above
x, y = EE_method(df_dx, y0, dx, x0, ns)

#Creates an x array for plotting the original function
x_old = linspace(x0, ns * dx + x0, ns)

#Creates a y array for plotting the original function
y_old = f(x_old)

#Plots the Explicit Euler (EE) method, the original function, and a legend
plt.plot(x, y, label = "EE")
plt.plot(x_old, y_old, label = "Original Calculation")
plt.legend(loc = "best")