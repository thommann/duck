import argparse

import duckdb

from matrix_to_table import matrix_to_table


def kronecker_to_db(input: str, input_a: str, input_b: str, db_name: str) -> None:
    """
    This function takes the name of the matrices, the matrices themselves, and the name of the database
    and inserts the matrices into the database.
    """
    name = input.split("/")[-1].split(".")[0]
    name_a = input_a.split("/")[-1].split(".")[0]
    name_b = input_b.split("/")[-1].split(".")[0]
    con = duckdb.connect(db_name)
    matrix_to_table(con, input, name)
    matrix_to_table(con, input_a, name_a)
    matrix_to_table(con, input_b, name_b)
    con.close()


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="CSV to DB converter")
    parser.add_argument("--input", type=str, required=True, help="Path to the original matrix CSV file")
    parser.add_argument("--input_a", type=str, required=True, help="Path to matrix A CSV file")
    parser.add_argument("--input_b", type=str, required=True, help="Path to matrix B CSV file")
    parser.add_argument("--database", type=str, required=True, help="Name of the database")
    args = vars(parser.parse_args())

    # input files must be valid CSV files
    if not args["input"].endswith(".csv"):
        raise ValueError("Original matrix input file is not a CSV file")
    if not args["input_a"].endswith(".csv"):
        raise ValueError("Matrix A input file is not a CSV file")
    if not args["input_b"].endswith(".csv"):
        raise ValueError("Matrix B input file is not a CSV file")

    return args


if __name__ == "__main__":
    args = parse_args()
    kronecker_to_db(args["input"], args["input_a"], args["input_b"], args["database"])
