import argparse

import duckdb

from matrix_to_table import matrix_to_table


def kronecker_to_db(name, name_a, name_b, input, input_a, input_b, db_name) -> None:
    """
    This function takes the name of the matrices, the matrices themselves, and the name of the database
    and inserts the matrices into the database.
    """
    con = duckdb.connect(db_name)
    matrix_to_table(con, input, name)
    matrix_to_table(con, input_a, name_a)
    matrix_to_table(con, input_b, name_b)
    con.close()


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="CSV to DB converter")
    parser.add_argument("--table", type=str, default="original", help="Name of the table for the original matrix")
    parser.add_argument("--table_a", type=str, default="matrix_a", help="Name of the table for matrix A")
    parser.add_argument("--table_b", type=str, default="matrix_b", help="Name of the table for matrix B")
    parser.add_argument("--input", type=str, required=True, help="Path to the original matrix CSV file")
    parser.add_argument("--input_a", type=str, required=True, help="Path to matrix A CSV file")
    parser.add_argument("--input_b", type=str, required=True, help="Path to matrix B CSV file")
    parser.add_argument("--database", type=str, required=True, help="Name of the database")
    args = vars(parser.parse_args())

    # Table names must be valid SQL identifiers
    if not args["table"].isidentifier():
        raise ValueError("Table name for the original matrix is not a valid SQL identifier")
    if not args["table_a"].isidentifier():
        raise ValueError("Table name for matrix A is not a valid SQL identifier")
    if not args["table_b"].isidentifier():
        raise ValueError("Table name for matrix B is not a valid SQL identifier")

    # Table names must be unique
    if args["table"] == args["table_a"]:
        raise ValueError("Table name for the original matrix is the same as the table name for matrix A")
    if args["table"] == args["table_b"]:
        raise ValueError("Table name for the original matrix is the same as the table name for matrix B")
    if args["table_a"] == args["table_b"]:
        raise ValueError("Table name for matrix A is the same as the table name for matrix B")

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
    kronecker_to_db(args["table"], args["table_a"], args["table_b"],
                    args["input"], args["input_a"], args["input_b"],
                    args["database"])
