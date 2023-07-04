import duckdb

from ML.params import middle_layer, mnist_matrices, iris_matrices, mnist_layers, iris_layers
from ML.extract_parameters import extract_parameters
from ML.calculate_kronecker import calculate_kronecker


def setup_db(model: str):
    matrices = iris_matrices if model == "iris" else mnist_matrices if model == "mnist" else None
    if matrices is None:
        raise ValueError(f"Invalid model name: {model}")

    con = duckdb.connect(f'data/{model}_{middle_layer[0]}x{middle_layer[1]}.db', read_only=False)
    for matrix in matrices:
        print(f"Processing {matrix}...", flush=True)
        table_name = matrix.replace('.', '_')
        filepath = f"data/{model}_{matrix}.csv"
        con.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS SELECT ROW_NUMBER() OVER () AS row_id, * FROM '{filepath}'")

    layers = iris_layers if model == "iris" else mnist_layers if model == "mnist" else None
    for layer in layers:
        table_name = layer[0].replace(".weight", "")
        print(f"Processing {table_name}...", flush=True)
        weights_file = f"data/{model}_{layer[0]}.csv"
        bias_file = f"data/{model}_{layer[1]}.csv"
        # Create weights table
        con.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS SELECT ROW_NUMBER() OVER () AS row_id, * FROM '{weights_file}'")
        # Add type column that will be 0 for weights and 1 for bias
        con.execute(f"ALTER TABLE {table_name} ADD COLUMN type INT DEFAULT 0")
        # Add bias to weights table
        con.execute(f"INSERT INTO {table_name} SELECT 0 AS row_idx, *, 1 AS type FROM '{bias_file}'")

    con.close()
    print("All done!")


def setup_iris():
    extract_parameters("iris")
    calculate_kronecker("iris")
    setup_db("iris")


def setup_mnist():
    extract_parameters("mnist")
    calculate_kronecker("mnist")
    setup_db("mnist")


def main():
    setup_iris()
    setup_mnist()


if __name__ == "__main__":
    main()
