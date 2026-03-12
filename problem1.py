import numpy as np
import matplotlib.pyplot as plt


# Solve the equation (to 4 significant digits) for n = 5, 10, 15, ... , 100 using one of the following methods: bisection, Newton, or secant. If initials are needed, 0 < x(n) < 3 can be used.
# I choose bisection method

# Fit the computed roots x(n) vs. n using the model function x(n) = 1 + a_1 ln n + a_2(ln n)^2 + a_3(ln n)^3

# (1) Using the Bisection Method
def f(x, n):
    return x**(x**x) - n # Subtracting the n to set equal to 0.

def bisection(n, tol=1e-6): # Have a tolerance greater than 4 significant figures to prevent rounding error, get to 4 significant figures later.
    a, b = 1.0, 3.0 # Initials provided
    while (b - a) / 2.0 > tol:
        c = (a + b) / 2.0
        if f(c, n) == 0:
            break
        elif f(a, n) * f(c, n) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2.0

# 5 through 100
n_vals = np.arange(5, 105, 5)
x_vals = np.array([bisection(n) for n in n_vals])

# (2) Fit the computed roots x(n) vs. n using the model function x(n) = 1 + a_1 ln n + a_2(ln n)^2 + a_3(ln n)^3

# y = a1*ln(n) + a2*(ln n)^2 + a3*(ln n)^3
y = x_vals - 1
ln_n = np.log(n_vals)

# (a) Design Matrix A  - The inconsistent overdetermined system of equations.
# Each column is one of the powers for the ln above
A = np.vstack([ln_n, ln_n**2, ln_n**3]).T

# (b) Normal Equations: (A^T * A) * a = A^T * y
M = A.T @ A # (A^T * A)
v = A.T @ y # A^T * y

# (c) Gaussian Elimination 
def gaussian_elimination(A_mat, b_vec):
    n = len(b_vec)
    # Augment matrix with the right-hand side vector
    aug = np.column_stack((A_mat, b_vec)).astype(float)
    
    # 1. Forward Elimination with Partial Pivoting
    for i in range(n):
        # Pivot: find the row with the largest absolute value in current column
        pivot_row = i + np.argmax(np.abs(aug[i:, i]))
        aug[[i, pivot_row]] = aug[[pivot_row, i]]
        
        for j in range(i + 1, n):
            factor = aug[j, i] / aug[i, i]
            aug[j, i:] -= factor * aug[i, i:]
            
    # 2. Back Substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1): # Working backwards by 1
        x[i] = (aug[i, -1] - np.dot(aug[i, i+1:n], x[i+1:n])) / aug[i, i]
    return x

a1, a2, a3 = gaussian_elimination(M, v)

print(f"Coefficients: a1={a1:.4f}, a2={a2:.4f}, a3={a3:.4f}") # Coefficients calculated with Gaussian Elimination

# Table from part 1
print("TABLE")
print(f"{'n':<5} | {'x(n)':<10}")
print("-" * 18)
for n, x in zip(n_vals, x_vals):
    print(f"{n:<5} | {x:.4f}")


# Plotting
plt.figure(figsize=(8, 5))
# Data for reference
plt.scatter(n_vals, x_vals, label='Computed Data', color='red')
n_dense = np.linspace(5, 100, 200)
# Uses the coefficients solved for and applies model function formula to fit 
x_fit = 1 + a1*np.log(n_dense) + a2*(np.log(n_dense)**2) + a3*(np.log(n_dense)**3)
plt.plot(n_dense, x_fit, label='Polynomial Fit', color='blue')
plt.xlabel('n')
plt.ylabel('x(n)')
plt.legend()
plt.grid(True)
plt.title('Roots x(n) vs. n')
plt.show()

