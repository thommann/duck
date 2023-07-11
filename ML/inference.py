import time

import duckdb
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch

from ML.params import middle_layer, use_sigmoid, iris_default_relations, mnist_default_relations, mnist_krone_relations, \
    iris_krone_relations, k, nr_runs, use_index_join


def insert(db_connection: duckdb.DuckDBPyConnection, h_table: str, h_vector: np.ndarray) -> str:
    db_connection.execute(
        f"CREATE OR REPLACE TABLE {h_table} (row INTEGER, col INTEGER, val DOUBLE)"
    )
    if use_index_join:
        # Create indices on row and col
        db_connection.execute(
            f"DROP INDEX IF EXISTS {h_table}_row_idx;"
            f"DROP INDEX IF EXISTS {h_table}_col_idx;"
            f"CREATE INDEX {h_table}_row_idx ON {h_table}(row);"
            f"CREATE INDEX {h_table}_col_idx ON {h_table}(col);"
        )
    row = 0
    for val in h_vector:
        db_connection.execute(f"INSERT INTO {h_table} VALUES ({row}, 0, {val})")
        row += 1
    db_connection.execute(f"INSERT INTO {h_table} VALUES ({row}, 0,  1)")
    return h_table


def matmul(a_table: str, b_table: str) -> str:
    # Matrix multiplication: AB
    return f"SELECT A.row, B.col, SUM(A.val * B.val) AS val FROM {a_table} A, {b_table} B " \
           f"WHERE A.col = B.row GROUP BY A.row, B.col"


def add_bias_neuron(z_table: str) -> str:
    # Add bias neuron
    return f"SELECT row, col, val FROM {z_table} UNION ALL SELECT MAX(row) + 1, 0, 1 FROM {z_table}"


def linear_default(w_table: str, h_table: str) -> str:
    # Matrix multiplication: WZ
    query = matmul(w_table, h_table)
    print(query)
    return query


def linear_krone(a_table: str, b_table: str, h_table: str) -> str:
    # Reshape H to (max(b_table.col) + 1, max(a_table.col) + 1)
    query = f"WITH\n" \
            f"M2 AS(SELECT MAX(row) + 1 AS val FROM {b_table}),\n" \
            f"N2 AS(SELECT MAX(col) + 1 AS val FROM {b_table}),\n" \
            f"VH AS(\n" \
            f"SELECT\n" \
            f"row % ((SELECT val FROM N2)) AS row,\n" \
            f"FLOOR(row / ((SELECT val FROM N2))) AS col,\n" \
            f"val FROM {h_table}\n" \
            f"),\n"
    for rank in range(k):
        query += f"B{rank} AS(SELECT * FROM {b_table} WHERE k = {rank}),\n"
        # Matrix multiplication: B(VH)
        query += f"BVH{rank} AS({matmul(f'B{rank}', 'VH')}),\n"
        query += f"A{rank} AS(SELECT * FROM {a_table} WHERE k = {rank}),\n"
        # Matrix multiplication: (BVH)A
        query += f"BVHA{rank} AS({matmul(f'BVH{rank}', f'A{rank}')}),\n"

    # Add all the BVHAs together
    values = [f"BVHA{rank}.val" for rank in range(k)]
    tables = [f"BVHA{rank}" for rank in range(k)]
    conditions = [f"BVHA{rank}.row = BVHA0.row AND BVHA{rank}.col = BVHA0.col" for rank in range(1, k)]
    query += f"BVHA AS (" \
             f"SELECT BVHA0.row, BVHA0.col, {' + '.join(values)} AS val\n" \
             f"FROM {' , '.join(tables)}\n"
    if len(conditions) > 0:
        query += f"WHERE {' AND '.join(conditions)}\n"
    query += f"ORDER BY BVHA0.row, BVHA0.col\n" \
             f")\n"

    # Reshape BVHA to ((max(a_table.row) + 1) * (max(b_table.row) + 1), 1)
    query += f"SELECT\n" \
             f"row + col * (SELECT val FROM M2) AS row,\n" \
             f"0 AS col,\n" \
             f"val FROM BVHA\n" \
             f"ORDER BY row\n"
    print(query)
    return query


def relu(z_table: str) -> str:
    return f"SELECT row, col, CASE WHEN val < 0 THEN 0 ELSE val END AS val FROM {z_table}"


def sigmoid(z_table: str) -> str:
    return f"SELECT row, col, 1 / (1 + EXP(-val)) AS val FROM {z_table}"


activation = sigmoid if use_sigmoid else relu


def softmax(z_table: str) -> str:
    return f"SELECT row, col, EXP(val) / SUM(EXP(val)) OVER () AS val FROM {z_table}"


def execute(db_connection: duckdb.DuckDBPyConnection, table: str, query: str) -> str:
    query = f"CREATE OR REPLACE TABLE {table} AS (\n{query}\n)"
    db_connection.execute(query)
    return table


def run_default(db_connection: duckdb.DuckDBPyConnection, x_vector: np.ndarray, model: str) -> str:
    relations = iris_default_relations if model == "iris" else mnist_default_relations if model == "mnist" else None
    if relations is None:
        raise ValueError(f"Unknown model {model}")

    # Load the input
    start = time.time()
    x = insert(db_connection, f"X", x_vector)
    print(f"Inserting X took {time.time() - start} seconds")

    # Inference query
    # FC1
    start = time.time()
    z1 = execute(db_connection, f"Z1", linear_default(relations[0], x))
    h1_ = execute(db_connection, f"H1_", activation(z1))
    h1 = execute(db_connection, f"H1", add_bias_neuron(h1_))
    print(f"FC1 took {time.time() - start} seconds")

    # FC2
    start = time.time()
    z2 = execute(db_connection, f"Z2", linear_default(relations[1], h1))
    h2_ = execute(db_connection, f"H2_", activation(z2))
    h2 = execute(db_connection, f"H2", add_bias_neuron(h2_))
    print(f"FC2 took {time.time() - start} seconds")

    # FC3
    start = time.time()
    z3 = execute(db_connection, f"Z3", linear_default(relations[2], h2))
    y = execute(db_connection, f"H3", softmax(z3))
    print(f"FC3 took {time.time() - start} seconds")

    return y


def run_krone(db_connection: duckdb.DuckDBPyConnection, x_vector: np.ndarray, model: str) -> str:
    relations = iris_krone_relations if model == "iris" else mnist_krone_relations if model == "mnist" else None
    if relations is None:
        raise ValueError(f"Unknown model {model}")

    # Load the input
    x = insert(db_connection, f"X", x_vector)

    # Inference query
    # FC1
    z1 = execute(db_connection, f"KDZ1", linear_krone(relations[0], relations[1], x))
    h1_ = execute(db_connection, f"KDH1_", activation(z1))
    h1 = execute(db_connection, f"KDH1", add_bias_neuron(h1_))

    # FC2
    z2 = execute(db_connection, f"KDZ2", linear_krone(relations[2], relations[3], h1))
    h2_ = execute(db_connection, f"KDH2_", activation(z2))
    h2 = execute(db_connection, f"KDH2", add_bias_neuron(h2_))

    # FC3
    z3 = execute(db_connection, f"KDZ3", linear_krone(relations[4], relations[5], h2))
    y = execute(db_connection, f"KDZ3", softmax(z3))

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
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Take first sample as test
    X_test = X[:1]
    y_test = y[:1]

    con = duckdb.connect(f'data/{model}_{middle_layer[0]}x{middle_layer[1]}.db', read_only=False)

    # Pragma settings
    con.execute(
        "PRAGMA threads=10;"
        "PRAGMA force_index_join;"
    )

    predictions_default = []
    predictions_krone = []
    times_default = []
    times_krone = []
    j = 0
    for x, y_0 in zip(X_test, y_test):
        j += 1
        print(f"Running inference for sample {j}/{len(X_test)}")
        # Run default
        y_table_default = run_default(con, x, model)
        y_vector_default = con.execute(
            f"SELECT val FROM {y_table_default} ORDER BY row, col").fetchall()
        y_predicted_default = np.argmax(y_vector_default)
        predictions_default.append(y_predicted_default)

        # Run Kronecker
        y_table_krone = run_krone(con, x, model)
        y_vector_krone = con.execute(
            f"SELECT val FROM {y_table_krone} ORDER BY row, col").fetchall()
        y_predicted_krone = np.argmax(y_vector_krone)
        predictions_krone.append(y_predicted_krone)

        # Time the queries
        for i in range(nr_runs):
            start = time.time()
            run_default(con, x, model)
            end = time.time()
            times_default.append(end - start)

        for i in range(nr_runs):
            start = time.time()
            run_krone(con, x, model)
            end = time.time()
            times_krone.append(end - start)

    con.close()
    print()

    # Print the accuracy
    print(f"Default accuracy: {np.mean(predictions_default == y_test):.2f}")
    print(f"Kronecker accuracy: {np.mean(predictions_krone == y_test):.2f}")

    # Print the times
    print(f"Default time: {np.mean(times_default) * 1000:.0f}ms")
    print(f"Kronecker time: {np.mean(times_krone) * 1000:.0f}ms")

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
