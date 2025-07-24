import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class CalculusToolkit:
    """
    A toolkit for understanding calculus concepts in machine learning
    """
    
    @staticmethod
    def numerical_derivative(f, x, h=1e-7):
        """
        Compute numerical derivative using finite differences
        """
        return (f(x + h) - f(x - h)) / (2 * h)
    
    @staticmethod
    def numerical_gradient(f, x, h=1e-7):
        """
        Compute numerical gradient for multivariable functions
        """
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
        return grad
    
    @staticmethod
    def visualize_derivative(f, f_prime, x_range=(-3, 3), x_point=1):
        """
        Visualize function and its derivative at a point
        """
        x = np.linspace(x_range[0], x_range[1], 1000)
        y = f(x)
        
        # Tangent line at x_point
        slope = f_prime(x_point)
        y_point = f(x_point)
        tangent_x = np.linspace(x_point - 1, x_point + 1, 100)
        tangent_y = y_point + slope * (tangent_x - x_point)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-', label=f'f(x)', linewidth=2)
        plt.plot(tangent_x, tangent_y, 'r--', label=f'Tangent at x={x_point}', linewidth=2)
        plt.plot(x_point, y_point, 'ro', markersize=8, label=f'Point ({x_point}, {y_point:.2f})')
        plt.grid(True, alpha=0.3)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'Function and Tangent Line (slope = {slope:.2f})')
        plt.legend()
        plt.show()

class GradientDescentOptimizer:
    """
    Basic gradient descent optimizer implementation
    """
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.history = []
    
    def minimize(self, f, gradient_f, initial_point, max_iterations=1000, tolerance=1e-6):
        """
        Minimize function using gradient descent
        """
        x = np.array(initial_point, dtype=float)
        self.history = [x.copy()]
        
        for i in range(max_iterations):
            grad = gradient_f(x)
            
            # Check for convergence
            if np.linalg.norm(grad) < tolerance:
                print(f"Converged after {i+1} iterations")
                break
            
            # Update parameters
            x = x - self.learning_rate * grad
            self.history.append(x.copy())
        
        return x, f(x)
    
    def visualize_optimization_2d(self, f, x_range=(-3, 3), y_range=(-3, 3)):
        """
        Visualize optimization path for 2D functions
        """
        if len(self.history) == 0:
            print("No optimization history found. Run minimize() first.")
            return
        
        # Create meshgrid for contour plot
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
        
        # Plot contours and optimization path
        plt.figure(figsize=(12, 5))
        
        # Contour plot
        plt.subplot(1, 2, 1)
        contours = plt.contour(X, Y, Z, levels=20, alpha=0.6)
        plt.contourf(X, Y, Z, levels=20, alpha=0.3, cmap='viridis')
        
        # Plot optimization path
        history = np.array(self.history)
        plt.plot(history[:, 0], history[:, 1], 'ro-', markersize=4, linewidth=2, 
                label='Optimization Path')
        plt.plot(history[0, 0], history[0, 1], 'go', markersize=8, label='Start')
        plt.plot(history[-1, 0], history[-1, 1], 'bo', markersize=8, label='End')
        
        plt.xlabel('x₁')
        plt.ylabel('x₂')
        plt.title('Optimization Path')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss curve
        plt.subplot(1, 2, 2)
        losses = [f(point) for point in self.history]
        plt.plot(losses, 'b-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss During Optimization')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Example functions for demonstration
def quadratic_1d(x):
    """Simple quadratic function: f(x) = x² - 4x + 3"""
    return x**2 - 4*x + 3

def quadratic_1d_derivative(x):
    """Derivative of quadratic function: f'(x) = 2x - 4"""
    return 2*x - 4

def quadratic_2d(x):
    """2D quadratic function: f(x,y) = x² + y² - 2x + 4y + 5"""
    return x[0]**2 + x[1]**2 - 2*x[0] + 4*x[1] + 5

def quadratic_2d_gradient(x):
    """Gradient of 2D quadratic function"""
    return np.array([2*x[0] - 2, 2*x[1] + 4])

def rosenbrock(x):
    """Rosenbrock function - classic optimization test case"""
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def rosenbrock_gradient(x):
    """Gradient of Rosenbrock function"""
    dx = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    dy = 200 * (x[1] - x[0]**2)
    return np.array([dx, dy])

# Chain rule demonstration
class ChainRuleDemo:
    """
    Demonstrate chain rule with neural network-like computations
    """
    
    @staticmethod
    def forward_and_backward():
        """
        Demonstrate forward pass and backpropagation using chain rule
        """
        print("Chain Rule Demo: Simple Neural Network Computation")
        print("=" * 50)
        
        # Forward pass: z = σ(wx + b) where σ is sigmoid
        w = 2.0
        x = 1.5
        b = 0.5
        
        # Step 1: Linear combination
        u = w * x + b  # u = 2.0 * 1.5 + 0.5 = 3.5
        print(f"Step 1: u = wx + b = {w} * {x} + {b} = {u}")
        
        # Step 2: Sigmoid activation
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))
        
        z = sigmoid(u)  # z = σ(3.5) ≈ 0.970
        print(f"Step 2: z = σ(u) = σ({u}) = {z:.6f}")
        
        # Backward pass: compute dz/dw using chain rule
        # dz/dw = dz/du * du/dw
        
        # dz/du = σ'(u) = σ(u) * (1 - σ(u))
        dz_du = z * (1 - z)
        print(f"\nBackward pass:")
        print(f"dz/du = σ'(u) = σ(u) * (1 - σ(u)) = {z:.6f} * {1-z:.6f} = {dz_du:.6f}")
        
        # du/dw = x
        du_dw = x
        print(f"du/dw = x = {du_dw}")
        
        # Chain rule: dz/dw = dz/du * du/dw
        dz_dw = dz_du * du_dw
        print(f"dz/dw = dz/du * du/dw = {dz_du:.6f} * {du_dw} = {dz_dw:.6f}")
        
        return dz_dw

# Demonstration and exercises
if __name__ == "__main__":
    print("Calculus Fundamentals for Machine Learning")
    print("=" * 40)
    
    # 1. Derivative visualization
    print("\n1. Visualizing Derivatives")
    calc_toolkit = CalculusToolkit()
    calc_toolkit.visualize_derivative(quadratic_1d, quadratic_1d_derivative, x_point=2)
    
    # 2. Numerical vs analytical derivatives
    print("\n2. Numerical vs Analytical Derivatives")
    x_test = 2.0
    analytical = quadratic_1d_derivative(x_test)
    numerical = calc_toolkit.numerical_derivative(quadratic_1d, x_test)
    print(f"At x = {x_test}:")
    print(f"Analytical derivative: {analytical}")
    print(f"Numerical derivative: {numerical:.8f}")
    print(f"Difference: {abs(analytical - numerical):.2e}")
    
    # 3. Gradient descent optimization
    print("\n3. Gradient Descent Optimization")
    
    # Simple 2D quadratic
    optimizer = GradientDescentOptimizer(learning_rate=0.1)
    result, final_loss = optimizer.minimize(
        quadratic_2d, 
        quadratic_2d_gradient, 
        initial_point=[3, -2],
        max_iterations=100
    )
    print(f"Optimization result: {result}")
    print(f"Final loss: {final_loss:.6f}")
    optimizer.visualize_optimization_2d(quadratic_2d)
    
    # 4. Chain rule demonstration
    print("\n4. Chain Rule Demonstration")
    chain_demo = ChainRuleDemo()
    gradient = chain_demo.forward_and_backward()
    
    # 5. Optimization landscape visualization
    print("\n5. Rosenbrock Function Optimization (Challenging Case)")
    optimizer_hard = GradientDescentOptimizer(learning_rate=0.001)
    result_hard, final_loss_hard = optimizer_hard.minimize(
        rosenbrock,
        rosenbrock_gradient,
        initial_point=[-1, 1],
        max_iterations=5000
    )
    print(f"Rosenbrock optimization result: {result_hard}")
    print(f"Final loss: {final_loss_hard:.6f}")
    print(f"True minimum is at (1, 1) with loss = 0")
    
    print("\nKey Takeaways:")
    print("- Derivatives measure rate of change")
    print("- Gradients point toward steepest increase")
    print("- Negative gradients guide us toward minima")
    print("- Chain rule enables backpropagation")
    print("- Different functions have different optimization challenges")