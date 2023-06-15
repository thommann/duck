import duckdb

from ML.params import middle_layer

matrices = [f'fc1.weight_4x{middle_layer[0]}', f'fc1.bias_{middle_layer[0]}x1',
            f'fc2.weight_{middle_layer[0]}x{middle_layer[1]}',
            f'fc2.bias_{middle_layer[1]}x1', f'fc3.weight_{middle_layer[1]}x3', f'fc3.bias_3x1']
kron_matrices = [f'fc1.weight_4x{middle_layer[0]}_a', f'fc1.weight_4x{middle_layer[0]}_b',
                 f'fc1.bias_{middle_layer[0]}x1_a',
                 f'fc1.bias_{middle_layer[0]}x1_b', f'fc2.weight_{middle_layer[0]}x{middle_layer[1]}_a',
                 f'fc2.weight_{middle_layer[0]}x{middle_layer[1]}_b', f'fc2.bias_{middle_layer[1]}x1_a',
                 f'fc2.bias_{middle_layer[1]}x1_b', f'fc3.weight_{middle_layer[1]}x3_a',
                 f'fc3.weight_{middle_layer[1]}x3_b',
                 f'fc3.bias_3x1_a', f'fc3.bias_3x1_b']

matrices += kron_matrices

con = duckdb.connect(f'ml{middle_layer[0]}x{middle_layer[1]}.db', read_only=False)
for matrix in matrices:
    print(f"Processing {matrix}...", end='\r')
    table_name = matrix.replace('.', '_')
    filepath = f"{matrix}.csv"
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT ROW_NUMBER() OVER () AS row_id, * FROM '{filepath}'")
    print(f"Processing {matrix}... Done!")

con.close()
print("All done!")
