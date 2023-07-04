middle_layer = (100, 50)
use_sigmoid = True

middle_layer_a = (10, 5)  # 100 x 50
middle_layer_b = (10, 10)  # 100 x 50

k = 3
max_k = 5

shape_a1 = (100, 2)  # (4, 100)^T
shape_a2 = (50, 10)  # (100, 50)^T
shape_a3 = (3, 5)  # (50, 3)^T

shape_b1 = (1, 2)  # (4, 100)^T
shape_b2 = (1, 10)  # (100, 50)^T
shape_b3 = (1, 10)  # (50, 3)^T

iris_weight_matrices = [f'fc1.weight_4x{middle_layer[0]}',
                        f'fc2.weight_{middle_layer[0]}x{middle_layer[1]}',
                        f'fc3.weight_{middle_layer[1]}x3']

iris_bias_matrices = [f'fc1.bias_{middle_layer[0]}x1',
                      f'fc2.bias_{middle_layer[1]}x1',
                      f'fc3.bias_3x1']

iris_bias_matrices_transpose = [f'fc1.bias_1x{middle_layer[0]}',
                                f'fc2.bias_1x{middle_layer[1]}',
                                f'fc3.bias_1x3']

iris_kron_matrices = [f'fc1.weight_4x{middle_layer[0]}_a',
                      f'fc1.weight_4x{middle_layer[0]}_b',
                      f'fc2.weight_{middle_layer[0]}x{middle_layer[1]}_a',
                      f'fc2.weight_{middle_layer[0]}x{middle_layer[1]}_b',
                      f'fc3.weight_{middle_layer[1]}x3_a',
                      f'fc3.weight_{middle_layer[1]}x3_b']

iris_matrices = iris_weight_matrices + iris_bias_matrices + iris_kron_matrices
iris_layers = zip(iris_weight_matrices, iris_bias_matrices_transpose)

mnist_weight_matrices = [f'fc1.weight_784x{middle_layer[0]}',
                         f'fc2.weight_{middle_layer[0]}x{middle_layer[1]}',
                         f'fc3.weight_{middle_layer[1]}x10']

mnist_bias_matrices = [f'fc1.bias_{middle_layer[0]}x1',
                       f'fc2.bias_{middle_layer[1]}x1',
                       f'fc3.bias_10x1']

mnist_bias_matrices_transpose = [f'fc1.bias_1x{middle_layer[0]}',
                                 f'fc2.bias_1x{middle_layer[1]}',
                                 f'fc3.bias_1x10']

mnist_kron_matrices = [f'fc1.weight_784x{middle_layer[0]}_a',
                       f'fc1.weight_784x{middle_layer[0]}_b',
                       f'fc2.weight_{middle_layer[0]}x{middle_layer[1]}_a',
                       f'fc2.weight_{middle_layer[0]}x{middle_layer[1]}_b',
                       f'fc3.weight_{middle_layer[1]}x10_a',
                       f'fc3.weight_{middle_layer[1]}x10_b']

mnist_matrices = mnist_weight_matrices + mnist_bias_matrices + mnist_kron_matrices
mnist_layers = zip(mnist_weight_matrices, mnist_bias_matrices_transpose)
