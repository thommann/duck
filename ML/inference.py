import time

import duckdb
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch

from ML.params import middle_layer, use_sigmoid, iris_default_relations, mnist_default_relations, mnist_krone_relations, \
    iris_krone_relations


def insert(db_connection: duckdb.DuckDBPyConnection, z_table: str, z_vector: np.ndarray) -> str:
    db_connection.execute(f"CREATE OR REPLACE TABLE {z_table} (row INTEGER, col INTEGER, val DOUBLE)")
    row = 0
    for val in z_vector:
        db_connection.execute(f"INSERT INTO {z_table} VALUES ({row}, 0, {val})")
        row += 1
    db_connection.execute(f"INSERT INTO {z_table} VALUES ({row + 1}, 0,  1)")
    return z_table


def matmul(a_table: str, b_table: str) -> str:
    # Matrix multiplication: AB
    return f"SELECT A.row, B.col, SUM(A.val * B.val) AS val FROM {a_table} A, {b_table} B " \
           f"WHERE A.col = B.row GROUP BY A.row, B.col"


def add_bias_neuron(z_table: str) -> str:
    # Add bias neuron
    return f"SELECT row, col, val FROM {z_table} UNION ALL SELECT MAX(row) + 1, 0, 1 FROM {z_table}"


def linear_default(w_table: str, z_table: str) -> str:
    # Matrix multiplication: WZ
    query = matmul(w_table, z_table)
    return query


def linear_krone(a_table: str, b_table: str, z_table: str) -> str:
    # Reshape Z to (max(b_table.col) + 1, max(a_table.col) + 1)
    query = f"WITH N2 AS(SELECT MAX(col) + 1 AS val FROM {b_table}), " \
            f"VZ AS(" \
            f"SELECT " \
            f"FLOOR(row / (SELECT val FROM N2)) AS row, " \
            f"row % (SELECT val FROM N2) AS col, " \
            f"val FROM {z_table}" \
            f"), "
    # Matrix multiplication: B(VZ)
    query += f"BVZ AS({matmul(b_table, 'VZ')}), "
    # Matrix multiplication: (BVZ)A
    query += f"BVZA AS({matmul('BVZ', a_table)}), "
    # Reshape BVHA to ((max(a_table.row) + 1) * (max(b_table.row) + 1), 1)
    query += f"M1 AS(SELECT MAX(col) + 1 AS val FROM {a_table}), " \
             f"M2 AS(SELECT MAX(row) + 1 AS val FROM {b_table}) " \
             f"SELECT " \
             f"row + col * (SELECT val FROM M2) AS row, " \
             f"0 AS col, " \
             f"val FROM BVZA " \
             f"ORDER BY row"
    return query


def relu(h_table: str) -> str:
    return f"SELECT row, col, CASE WHEN val < 0 THEN 0 ELSE val END AS val FROM {h_table}"


def sigmoid(h_table: str) -> str:
    return f"SELECT row, col, 1 / (1 + EXP(-val)) AS val FROM {h_table}"


activation = sigmoid if use_sigmoid else relu


def softmax(h_table: str) -> str:
    return f"SELECT row, col, EXP(val) / SUM(EXP(val)) OVER () AS val FROM {h_table}"


def execute(db_connection: duckdb.DuckDBPyConnection, table: str, query: str) -> str:
    query = f"CREATE OR REPLACE TABLE {table} AS (\n{query}\n)"
    db_connection.execute(query)
    return table


def run_default(db_connection: duckdb.DuckDBPyConnection, x_vector: np.ndarray, model: str) -> str:
    relations = iris_default_relations if model == "iris" else mnist_default_relations if model == "mnist" else None
    if relations is None:
        raise ValueError(f"Unknown model {model}")

    # Load the input
    x = insert(db_connection, f"X", x_vector)

    # Inference query
    # FC1
    h1_ = execute(db_connection, f"H1_", linear_default(relations[0], x))
    h1 = execute(db_connection, f"H1", add_bias_neuron(h1_))
    z1 = execute(db_connection, f"Z1", activation(h1))

    # FC2
    h2_ = execute(db_connection, f"H2_", linear_default(relations[1], z1))
    h2 = execute(db_connection, f"H2", add_bias_neuron(h2_))
    z2 = execute(db_connection, f"Z2", activation(h2))

    # FC3
    h3_ = execute(db_connection, f"H3_", linear_default(relations[2], z2))
    y = execute(db_connection, f"Z3", softmax(h3_))

    return y


def run_krone(db_connection: duckdb.DuckDBPyConnection, x_vector: np.ndarray, model: str) -> str:
    relations = iris_krone_relations if model == "iris" else mnist_krone_relations if model == "mnist" else None
    if relations is None:
        raise ValueError(f"Unknown model {model}")

    # Load the input
    x = insert(db_connection, f"X", x_vector)

    # Inference query
    # FC1
    h1_ = execute(db_connection, f"KDH1_", linear_krone(relations[0], relations[1], x))
    h1 = execute(db_connection, f"KDH1", add_bias_neuron(h1_))
    z1 = execute(db_connection, f"KDZ1", activation(h1))

    # FC2
    h2_ = execute(db_connection, f"KDH2_", linear_krone(relations[2], relations[3], z1))
    h2 = execute(db_connection, f"KDH2", add_bias_neuron(h2_))
    z2 = execute(db_connection, f"KDZ2", activation(h2))

    # FC3
    h3_ = execute(db_connection, f"KDH3_", linear_krone(relations[4], relations[5], z2))
    y = execute(db_connection, f"KDZ3", softmax(h3_))

    return y


def inference(dataset: Bunch, model: str):
    # Load the dataset
    X = dataset.data
    y = dataset.target
    y = y.astype(int)

    # Scale the features for better training
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Take first sample as test
    # X_test = X[:1]
    # y_test = y[:1]

    con = duckdb.connect(f'data/{model}_{middle_layer[0]}x{middle_layer[1]}.db', read_only=False)

    predictions_default = []
    predictions_krone = []
    for x, y_0 in zip(X_test, y_test):
        # Run default
        y_table_default = run_default(con, x, model)
        y_vector_default = con.execute(
            f"SELECT val FROM {y_table_default} ORDER BY row, col").fetchall()
        y_predicted_default = np.argmax(y_vector_default)
        predictions_default.append(y_predicted_default)
        print(f"Default: {y_predicted_default} (expected {y_0})")

        # Run Kronecker
        y_table_krone = run_krone(con, x, model)
        y_vector_krone = con.execute(
            f"SELECT val FROM {y_table_krone} ORDER BY row, col").fetchall()
        y_predicted_krone = np.argmax(y_vector_krone)
        predictions_krone.append(y_predicted_krone)
        print(f"Kronecker: {y_predicted_krone} (expected {y_0})")

    con.close()

    # Print the accuracy
    print(f"Default accuracy: {np.mean(predictions_default == y_test)}")
    print(f"Kronecker accuracy: {np.mean(predictions_krone == y_test)}")

    print("Done!")


def inference_iris():
    iris = datasets.load_iris()
    inference(iris, "iris")


def inference_mnist():
    mnist = datasets.fetch_openml('mnist_784', version=1, cache=True, as_frame=False, parser='liac-arff')
    inference(mnist, "mnist")


if __name__ == "__main__":
    inference_iris()
    # inference_mnist()
