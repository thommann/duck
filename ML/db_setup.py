import duckdb

from ML.params import middle_layer, matrices, iris_relations, mnist_relations, krone_matrices, use_index_join
from ML.extract_parameters import extract_parameters


def setup_db(model: str):
    relations = iris_relations if model == "iris" else mnist_relations if model == "mnist" else None
    if relations is None:
        raise ValueError(f"Unknown model: {model}")

    con = duckdb.connect(f'data/{model}_{middle_layer[0]}x{middle_layer[1]}.db', read_only=False)
    for matrix, relation in zip(matrices, relations):
        print(f"Processing {matrix}...", flush=True)
        filepath = f"data/{model}_{matrix}.csv"
        # Drop all indices of the relation
        con.execute(
            f"DROP INDEX IF EXISTS {relation}_k_idx;"
            f"DROP INDEX IF EXISTS {relation}_row_col_idx;"
            f"DROP INDEX IF EXISTS {relation}_row_col_k_idx;"
            f"DROP INDEX IF EXISTS {relation}_row_idx;"
            f"DROP INDEX IF EXISTS {relation}_col_idx;"
        )
        # Import the matrix
        con.execute(
            f"CREATE OR REPLACE TABLE {relation} AS SELECT * FROM '{filepath}'"
        )
        # Change the type of row to INTEGER (it's imported as DOUBLE by default)
        con.execute(
            f"ALTER TABLE {relation} "
            f"ALTER COLUMN row TYPE INTEGER"
        )
        # Change the type of col to INTEGER (it's imported as DOUBLE by default)
        con.execute(
            f"ALTER TABLE {relation} "
            f"ALTER COLUMN col TYPE INTEGER"
        )
        if matrix in krone_matrices:
            # Change the type of k to INTEGER (it's imported as DOUBLE by default)
            con.execute(
                f"ALTER TABLE {relation} "
                f"ALTER COLUMN k TYPE INTEGER"
            )
        if use_index_join:
            # Create the indices
            create_indices(con, matrix, relation)

    con.close()
    print("All done!")


def create_indices(con: duckdb.DuckDBPyConnection, matrix: str, relation: str):
    con.execute(
        f"CREATE INDEX {relation}_row_idx ON {relation}(row);"
        f"CREATE INDEX {relation}_col_idx ON {relation}(col);"
    )
    if matrix in krone_matrices:
        con.execute(
            f"CREATE INDEX {relation}_k_idx ON {relation}(k);"
        )


def setup_iris():
    extract_parameters("iris")
    setup_db("iris")


def setup_mnist():
    extract_parameters("mnist")
    setup_db("mnist")


def main():
    setup_iris()
    # setup_mnist()


if __name__ == "__main__":
    main()
