import numpy as np
import pandas as pd
from numpy.linalg import multi_dot

# Given values mu, S and R. To transform those vectors/matrix into arrays, we use np.array.
mu = np.array([0.02, 0.07, 0.15, 0.2])

S = np.array([0.05, 0.12, 0.17, 0.25])

R = np.array([[1, 0.3, 0.3, 0.3],
              [0.3, 1, 0.6, 0.6],
              [0.3, 0.6, 1, 0.6],
              [0.3, 0.6, 0.6, 1]])

# Covariance matrix using @ that internally handles vectors transpositions to perform multiplications with matrix.
cov_matrix = np.diag(S) @ R @ np.diag(S)

# Inverse of covariance matrix (precision matrix)
Sigma_inv = np.linalg.inv(cov_matrix)

# Vector of ones of lenght mu to fit the dimension of the matrix to make multiplications possible.
ones = np.ones(len(mu))              

# Compute A, B, and C
A = ones @ Sigma_inv @ ones
B = mu @ Sigma_inv @ ones
C = mu @ Sigma_inv @ mu

# Ensure AC - B^2 is greater than zero
det = A * C - B**2
if det <= 0:
    raise ValueError("AC - B^2 must be greater than 0 for the equation to be solvable.")

# Set m to 4.5%
m = 0.045

# Compute w*
w_star = (1 / det) * np.dot(Sigma_inv, (m * (A * mu - B * ones) + C * ones - B * mu))

print("Optimal w*:", w_star)

# Compute the portfolio variance
portfolio_variance = w_star @ cov_matrix @ w_star

# Compute the standard deviation (SD)
portfolio_sd = np.sqrt(portfolio_variance)

# Print the standard deviation
print("Portfolio Standard Deviation (SD):", portfolio_sd)
