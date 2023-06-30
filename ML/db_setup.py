import duckdb

from ML.params import middle_layer
from ML.extract_parameters import extract_parameters
from ML.calculate_kronecker import calculate_kronecker


def setup_db():
    matrices = [f'fc1.weight_4x{middle_layer[0]}',
                f'fc1.bias_{middle_layer[0]}x1',
                f'fc2.weight_{middle_layer[0]}x{middle_layer[1]}',
                f'fc2.bias_{middle_layer[1]}x1',
                f'fc3.weight_{middle_layer[1]}x3',
                f'fc3.bias_3x1']

    kron_matrices = [f'fc1.weight_4x{middle_layer[0]}_a',
                     f'fc1.weight_4x{middle_layer[0]}_b',
                     f'fc2.weight_{middle_layer[0]}x{middle_layer[1]}_a',
                     f'fc2.weight_{middle_layer[0]}x{middle_layer[1]}_b',
                     f'fc3.weight_{middle_layer[1]}x3_a',
                     f'fc3.weight_{middle_layer[1]}x3_b']

    matrices += kron_matrices

    con = duckdb.connect(f'data/ml{middle_layer[0]}x{middle_layer[1]}.db', read_only=False)
    for matrix in matrices:
        print(f"Processing {matrix}...", flush=True)
        table_name = matrix.replace('.', '_')
        filepath = f"data/{matrix}.csv"
        con.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS SELECT ROW_NUMBER() OVER () AS row_id, * FROM '{filepath}'")

    layers = [[f'fc1.weight_4x{middle_layer[0]}', f'fc1.bias_1x{middle_layer[0]}'],
              [f'fc2.weight_{middle_layer[0]}x{middle_layer[1]}', f'fc2.bias_1x{middle_layer[1]}'],
              [f'fc3.weight_{middle_layer[1]}x3', f'fc3.bias_1x3']]

    for layer in layers:
        table_name = layer[0].replace(".weight", "")
        print(f"Processing {table_name}...", flush=True)
        weights_file = f"data/{layer[0]}.csv"
        bias_file = f"data/{layer[1]}.csv"
        # Create weights table
        con.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS SELECT ROW_NUMBER() OVER () AS row_id, * FROM '{weights_file}'")
        # Add type column that will be 0 for weights and 1 for bias
        con.execute(f"ALTER TABLE {table_name} ADD COLUMN type INT DEFAULT 0")
        # Add bias to weights table
        con.execute(f"INSERT INTO {table_name} SELECT 0 AS row_idx, *, 1 AS type FROM '{bias_file}'")

    con.close()
    print("All done!")


if __name__ == "__main__":
    extract_parameters()
    calculate_kronecker()
    setup_db()
