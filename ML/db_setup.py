import duckdb
import torch

matrices = ['fc1.weight_4x100', 'fc1.bias_100x1', 'fc2.weight_100x50', 'fc2.bias_50x1', 'fc3.weight_50x3', 'fc3.bias_3x1']
kron_matrices = ['fc1.weight_4x100_a', 'fc1.weight_4x100_b', 'fc1.bias_100x1_a', 'fc1.bias_100x1_b',
                    'fc2.weight_100x50_a', 'fc2.weight_100x50_b', 'fc2.bias_50x1_a', 'fc2.bias_50x1_b',
                    'fc3.weight_50x3_a', 'fc3.weight_50x3_b', 'fc3.bias_3x1_a', 'fc3.bias_3x1_b']

matrices += kron_matrices

con = duckdb.connect('lm.db', read_only=False)
for matrix in matrices:
    table_name = matrix.replace('.', '_')
    filepath = f"{matrix}.csv"
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(f"CREATE TABLE {table_name} AS SELECT ROW_NUMBER() OVER () AS row_id, * FROM '{filepath}'")


con.close()
print("Done!")
