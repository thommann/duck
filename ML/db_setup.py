import duckdb

from ML.params import middle_layer, matrices, iris_relations, mnist_relations, krone_matrices
from ML.extract_parameters import extract_parameters


def setup_db(model: str):
    relations = iris_relations if model == "iris" else mnist_relations if model == "mnist" else None
    if relations is None:
        raise ValueError(f"Unknown model: {model}")

    con = duckdb.connect(f'data/{model}_{middle_layer[0]}x{middle_layer[1]}.db', read_only=False)
    for matrix, relation in zip(matrices, relations):
        print(f"Processing {matrix}...", flush=True)
        filepath = f"data/{model}_{matrix}.csv"
        con.execute(
            f"CREATE OR REPLACE TABLE {relation} AS SELECT * FROM '{filepath}'"
        )
        con.execute(
            f"ALTER TABLE {relation} "
            f"ALTER COLUMN row TYPE INTEGER"
        )
        con.execute(
            f"ALTER TABLE {relation} "
            f"ALTER COLUMN col TYPE INTEGER"
        )
        if matrix in krone_matrices:
            con.execute(
                f"ALTER TABLE {relation} "
                f"ALTER COLUMN k TYPE INTEGER"
            )
    con.close()
    print("All done!")


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
