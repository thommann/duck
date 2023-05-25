import duckdb


def csv_to_table(connection: duckdb.DuckDBPyConnection, filepath: str, name: str) -> None:
    connection.sql(f"DROP TABLE {name}")
    connection.sql(f"CREATE TABLE {name} AS SELECT * FROM '{filepath}'")


if __name__ == '__main__':
    connection = duckdb.connect("kronecker.db")
    connection.sql(f"SELECT * FROM test").show()
