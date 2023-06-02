import argparse

import duckdb

from src.matrix_to_table import matrix_to_table


def kronecker_to_db(input: str, input_a: str, input_b: str, database: str, k: int = 1) -> None:
    """
    This function takes the name of the matrices, the matrices themselves, and the name of the database
    and inserts the matrices into the database.
    """
    name_a, name_b, name = "A", "B", "C"
    con = duckdb.connect(database)
    matrix_to_table(con, input, name)
    if k == 1:
        matrix_to_table(con, input_a, name_a)
        matrix_to_table(con, input_b, name_b)
    else:
        for rank in range(1, k + 1):
            matrix_to_table(con, input_a.replace('.csv', f'_{rank}.csv'), name_a + f'_{rank}')
            matrix_to_table(con, input_b.replace('.csv', f'_{rank}.csv'), name_b + f'_{rank}')
    con.close()


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="CSV to DB converter")
    parser.add_argument("--input", type=str, required=True, help="Path to the original matrix CSV file")
    parser.add_argument("--input_a", type=str, required=True, help="Path to matrix A CSV file")
    parser.add_argument("--input_b", type=str, required=True, help="Path to matrix B CSV file")
    parser.add_argument("--database", type=str, required=True, help="Path to the database file")
    parser.add_argument("--rank", "-r", "-k", type=int, default=1, help="Rank of the Kronecker decomposition")
    args = vars(parser.parse_args())

    # input files must be valid CSV files
    if not args["input"].endswith(".csv"):
        raise ValueError("Original matrix input file is not a CSV file")
    if not args["input_a"].endswith(".csv"):
        raise ValueError("Matrix A input file is not a CSV file")
    if not args["input_b"].endswith(".csv"):
        raise ValueError("Matrix B input file is not a CSV file")

    # rank must be a positive integer
    if args["rank"] < 1:
        raise ValueError("Rank must be a positive integer")

    # database file must be a valid DuckDB database file
    if not args["database"].endswith(".db"):
        raise ValueError("Database file is not a DuckDB database file")

    return args


if __name__ == "__main__":
    args = parse_args()
    kronecker_to_db(args["input"], args["input_a"], args["input_b"], args["database"], k=args["rank"])
