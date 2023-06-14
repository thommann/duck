import numpy as np

from src.kronecker import svd, compute_shapes, kronecker_decomposition

matrices = ['fc1.weight_4x100', 'fc1.bias_100x1', 'fc2.weight_100x50', 'fc2.bias_50x1', 'fc3.weight_50x3', 'fc3.bias_3x1']

for matrix in matrices:
    filepath = f"{matrix}.csv"
    c = np.loadtxt(filepath, delimiter=',')
    c = np.atleast_2d(c.T).T
    shape_c = c.shape
    shape_a, shape_b = compute_shapes(shape_c)
    u, s, vh = svd(c, shape_a)
    a, b = kronecker_decomposition(u, s, vh, shape_a, shape_b, k=1)
    np.savetxt(f"{matrix}_a.csv", a, delimiter=',')
    np.savetxt(f"{matrix}_b.csv", b, delimiter=',')

print("Done!")
