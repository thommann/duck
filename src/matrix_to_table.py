import argparse

import duckdb


def matrix_to_table(connection: duckdb.DuckDBPyConnection, filepath: str, name: str) -> None:
    connection.execute(f"DROP TABLE IF EXISTS {name}")
    connection.execute(f"""
    CREATE TABLE {name} AS 
    WITH temp AS (
        SELECT * FROM '{filepath}'
    )
    SELECT ROW_NUMBER() OVER () AS id, * 
    FROM temp;
    """)


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="CSV matrix to table in database")
    parser.add_argument('--input', type=str, required=True, help="Path to input CSV file")
    parser.add_argument('--table', type=str, required=True, help="Name of the table")
    parser.add_argument('--database', type=str, required=True, help="Path to database file")
    args = vars(parser.parse_args())

    # Input must be a CSV file
    if not args['input'].endswith('.csv'):
        raise ValueError("Input must be a CSV file")

    # Table name must be a valid SQL identifier
    if not args['table'].isidentifier():
        raise ValueError("Table name must be a valid SQL identifier")

    return args


if __name__ == '__main__':
    args = parse_args()
    connection = duckdb.connect(args['database'])
    matrix_to_table(connection, args['input'], args['table'])
