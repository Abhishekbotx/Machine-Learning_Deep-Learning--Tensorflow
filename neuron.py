import numpy as np
def neuron(inputs,weight,bias):
    return np.dot(inputs,weight)+ bias

# np.dot() calculates the dot product of two vectors (or matrices).
# 1.5 * 0.2  = 0.3
# 2.0 * 0.8  = 1.6
# 3.0 * -0.5 = -1.5
# ------------------
# Total       = 0.4
# Dot product = weighted sum of inputs
# This is exactly what a neuron does before applying activation:
#     a1*b1 + a2*b2 + ... + an*bn

inputs=np.array([1.5,2.0,3.0])
weights = np.array([0.2, 0.8, -0.5])
bias=2.0

output = neuron(inputs, weights, bias)
print(f"Neuron output: {output}")