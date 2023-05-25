import argparse

import duckdb


def csv_to_table(connection: duckdb.DuckDBPyConnection, filepath: str, name: str) -> None:
    connection.sql(f"DROP TABLE IF EXISTS {name}")
    connection.sql(f"CREATE TABLE {name} AS SELECT * FROM '{filepath}'")


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="CSV matrix to table in database")
    parser.add_argument('--input', type=str, required=True, help="Path to input CSV file")
    parser.add_argument('--table', type=str, required=True, help="Name of the table")
    parser.add_argument('--database', type=str, required=True, help="Path to database file")
    args = vars(parser.parse_args())

    # Input must be a CSV file
    if not args['input'].endswith('.csv'):
        raise ValueError("Input must be a CSV file")

    return args


if __name__ == '__main__':
    args = parse_args()
    input = args['input']
    table = args['table']
    database = args['database']
    connection = duckdb.connect(args['database'])
    csv_to_table(connection, input, table)
