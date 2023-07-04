middle_layer = (100, 50)
use_sigmoid = True

middle_layer_a = (10, 5)  # 100 x 50
middle_layer_b = (10, 10)  # 100 x 50

k = 5
max_k = 5

iris_shape_a1 = (100, 2)  # (4, 100)^T
iris_shape_a2 = (50, 10)  # (100, 50)^T
iris_shape_a3 = (3, 5)  # (50, 3)^T

iris_shape_b1 = (1, 2)  # (4, 100)^T
iris_shape_b2 = (1, 10)  # (100, 50)^T
iris_shape_b3 = (1, 10)  # (50, 3)^T

iris_shapes = [[iris_shape_a1, iris_shape_b1],
               [iris_shape_a2, iris_shape_b2],
               [iris_shape_a3, iris_shape_b3]]

mnist_shape_a1 = (100, 28)  # (784, 100)^T
mnist_shape_a2 = (50, 10)  # (100, 50)^T
mnist_shape_a3 = (10, 5)  # (50, 10)^T

mnist_shape_b1 = (1, 28)  # (784, 100)^T
mnist_shape_b2 = (1, 10)  # (100, 50)^T
mnist_shape_b3 = (1, 10)  # (50, 10)^T

mnist_shapes = [[mnist_shape_a1, mnist_shape_b1],
                [mnist_shape_a2, mnist_shape_b2],
                [mnist_shape_a3, mnist_shape_b3]]

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

iris_default_relations = [[f'fc1_weight_4x{middle_layer[0]}', f'fc1_bias_{middle_layer[0]}x1'],
                          [f'fc2_weight_{middle_layer[0]}x{middle_layer[1]}', f'fc2_bias_{middle_layer[1]}x1'],
                          [f'fc3_weight_{middle_layer[1]}x3', f'fc3_bias_3x1']]

iris_alt_relations = [f'fc1_4x{middle_layer[0]}', f'fc2_{middle_layer[0]}x{middle_layer[1]}',
                      f'fc3_{middle_layer[1]}x3']

iris_krone_relations = [[f'fc1_weight_4x{middle_layer[0]}_a',
                         f'fc1_weight_4x{middle_layer[0]}_b',
                         f'fc1_bias_{middle_layer[0]}x1'],
                        [f'fc2_weight_{middle_layer[0]}x{middle_layer[1]}_a',
                         f'fc2_weight_{middle_layer[0]}x{middle_layer[1]}_b',
                         f'fc2_bias_{middle_layer[1]}x1'],
                        [f'fc3_weight_{middle_layer[1]}x3_a',
                         f'fc3_weight_{middle_layer[1]}x3_b',
                         f'fc3_bias_3x1']]

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

mnist_default_relations = [[f'fc1_weight_784x{middle_layer[0]}', f'fc1_bias_{middle_layer[0]}x1'],
                           [f'fc2_weight_{middle_layer[0]}x{middle_layer[1]}', f'fc2_bias_{middle_layer[1]}x1'],
                           [f'fc3_weight_{middle_layer[1]}x10', f'fc3_bias_10x1']]

mnist_alt_relations = [f'fc1_784x{middle_layer[0]}', f'fc2_{middle_layer[0]}x{middle_layer[1]}',
                       f'fc3_{middle_layer[1]}x10']

mnist_krone_relations = [[f'fc1_weight_784x{middle_layer[0]}_a',
                          f'fc1_weight_784x{middle_layer[0]}_b',
                          f'fc1_bias_{middle_layer[0]}x1'],
                         [f'fc2_weight_{middle_layer[0]}x{middle_layer[1]}_a',
                          f'fc2_weight_{middle_layer[0]}x{middle_layer[1]}_b',
                          f'fc2_bias_{middle_layer[1]}x1'],
                         [f'fc3_weight_{middle_layer[1]}x10_a',
                          f'fc3_weight_{middle_layer[1]}x10_b',
                          f'fc3_bias_10x1']]
