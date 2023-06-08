import numpy as np

mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

product = np.prod(mat, axis=1)
target = np.sum(product)

result = np.einsum('ij->i', mat).prod()

print(product)
print(target)
print()
print(result)
