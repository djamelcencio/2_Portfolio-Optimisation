import numpy as np
from scipy.stats import norm

mu = np.array([0, 0, 0])
S = np.array([0.3, 0.2, 0.15])
R = np.array([[1, 0.8, 0.5],
              [0.8, 1, 0.3],
              [0.5, 0.3, 1]])
weights_sets = np.array([0.5, 0.2, 0.3])

cov_matrix = np.diag(S) @ R @ np.diag(S)

alpha = 0.99
factor = norm.ppf(1 - alpha)

# Compute the denominator 
vector_computation = np.sqrt(weights_sets @ cov_matrix @ weights_sets)

# Compute VaR sensitivity with respect to each asset weight
var_sens = []

for i in range(len(weights_sets)):
    numerator = factor * np.sum(cov_matrix[i] * weights_sets[i])
    sensitivity = numerator / vector_computation
    var_sens.append(sensitivity)

print("VaR Sensitivities with respect to each asset weight:")
for i, sensitivity in enumerate(var_sens):
    print(f"Asset {i+1}: {sensitivity}")
    
# Compute ES sensitivity with respect to each asset weight

# Compute the denominator of the sensitivity
denominator = np.sqrt(np.dot(weights_sets, np.dot(cov_matrix, weights_sets)))

# Compute ES sensitivity with respect to each asset weight
es_sensitivities = []
for i in range(len(weights_sets)):
    numerator = factor * np.exp(norm.ppf(1 - alpha) ** 2 / 2) * np.sqrt(2 / np.pi) * S[i]
    sensitivity = numerator / denominator
    es_sensitivities.append(sensitivity)

print("ES Sensitivities with respect to each asset weight:")
for i, sensitivity in enumerate(es_sensitivities):
    print(f"Asset {i+1}: {sensitivity}")

# Create a DataFrame to store the results
data = {
    "Asset": np.arange(1, len(weights_sets)+1),
    "VaR Sensitivity": var_sens,
    "ES Sensitivity": es_sensitivities
}
df = pd.DataFrame(data)

# Display the table
df
