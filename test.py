import numpy as np

# Generate the 100x100 sequence
sequence = np.arange(14, 14 + 100 * 100 * 11, 11).reshape(100, 100)

# Extract the first row, first 10 columns
subset = sequence[0, :10]

# Compute the sum
result = np.sum(subset)

print("Sum:", result)
