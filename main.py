import duckdb
import numpy as np

from kronecker import kronecker_decomposition
from svd import SVD


def create_matrix_table(matrix: np.ndarray, connection: duckdb.DuckDBPyConnection, name: str) -> None:
    creation_query = f"CREATE TABLE {name} ("
    for col in range(matrix.shape[1]):
        creation_query += f"col{col} FLOAT, "
    creation_query = creation_query[:-2] + ")"

    connection.execute(creation_query)

    insert_query = f"INSERT INTO {name} VALUES "
    for row in range(matrix.shape[0]):
        insert_query += "("
        for col in range(matrix.shape[1]):
            insert_query += f"{matrix[row, col]}, "
        insert_query = insert_query[:-2] + "), "
    insert_query = insert_query[:-2]

    connection.execute(insert_query)


def create_kronecker_tables(svd: SVD, connection: duckdb.DuckDBPyConnection, name: str) -> None:
    a_mat, b_mat = kronecker_decomposition(svd)
