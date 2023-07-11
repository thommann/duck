middle_layer = (1000, 1000)
use_sigmoid = False

k = 4
max_k = 10

nr_runs = 1
use_index_join = True

fc_layers = [('fc1.weight', 'fc1.bias'),
             ('fc2.weight', 'fc2.bias'),
             ('fc3.weight', 'fc3.bias')]

default_matrices = [f'fc1_{middle_layer[0]}x{middle_layer[1]}',
                    f'fc2_{middle_layer[0]}x{middle_layer[1]}',
                    f'fc3_{middle_layer[0]}x{middle_layer[1]}']

krone_matrices = [f'fc1_{middle_layer[0]}x{middle_layer[1]}_a',
                  f'fc1_{middle_layer[0]}x{middle_layer[1]}_b',
                  f'fc2_{middle_layer[0]}x{middle_layer[1]}_a',
                  f'fc2_{middle_layer[0]}x{middle_layer[1]}_b',
                  f'fc3_{middle_layer[0]}x{middle_layer[1]}_a',
                  f'fc3_{middle_layer[0]}x{middle_layer[1]}_b']

matrices = default_matrices + krone_matrices

iris_default_relations = [f'fc1_4x{middle_layer[0]}',
                          f'fc2_{middle_layer[0]}x{middle_layer[1]}',
                          f'fc3_{middle_layer[1]}x3']

iris_krone_relations = [f'fc1_4x{middle_layer[0]}_a',
                        f'fc1_4x{middle_layer[0]}_b',
                        f'fc2_{middle_layer[0]}x{middle_layer[1]}_a',
                        f'fc2_{middle_layer[0]}x{middle_layer[1]}_b',
                        f'fc3_{middle_layer[1]}x3_a',
                        f'fc3_{middle_layer[1]}x3_b']

iris_relations = iris_default_relations + iris_krone_relations

mnist_default_relations = [f'fc1_784x{middle_layer[0]}',
                           f'fc2_{middle_layer[0]}x{middle_layer[1]}',
                           f'fc3_{middle_layer[1]}x10']

mnist_krone_relations = [f'fc1_784x{middle_layer[0]}_a',
                         f'fc1_784x{middle_layer[0]}_b',
                         f'fc2_{middle_layer[0]}x{middle_layer[1]}_a',
                         f'fc2_{middle_layer[0]}x{middle_layer[1]}_b',
                         f'fc3_{middle_layer[1]}x10_a',
                         f'fc3_{middle_layer[1]}x10_b']

mnist_relations = mnist_default_relations + mnist_krone_relations
