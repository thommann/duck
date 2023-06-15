import time

import duckdb
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from ML.calculate_kronecker import calculate_kronecker
from src.queries import kronecker_sum_product


def insert(con: duckdb.DuckDBPyConnection, table: str, x: np.ndarray) -> str:
    con.execute(f"CREATE OR REPLACE TABLE {table} (row_id INTEGER, value DOUBLE)")
    for i, val in enumerate(x):
        con.execute(f"INSERT INTO {table} VALUES ({i + 1}, {val})")
    return table


def execute(con: duckdb.DuckDBPyConnection, table: str, query: str) -> str:
    query = f"CREATE OR REPLACE TABLE {table} AS (\n{query}\n)"
    con.execute(query)
    return table


def linear(cols: int, z_relation: str, w_relation: str, b_relation: str) -> str:
    query = f"""WITH
combo AS (
SELECT * FROM {z_relation} z, {w_relation} w WHERE z.row_id = w.row_id
),
a AS (
"""
    terms = []
    for i in range(cols):
        terms.append(f"SELECT {i + 1} AS row_id, SUM(c.value * c.column{i:{len(str(int(cols - 1))):02d}d}) AS value "
                     f"FROM combo c")
    query += "\nUNION ALL\n".join(terms)
    query += "\n)\n"
    query += f"SELECT a.row_id, a.value + b.column0 AS value FROM a, {b_relation} b WHERE a.row_id = b.row_id"
    return query


def relu(h_relation: str) -> str:
    return f"SELECT row_id, CASE WHEN value < 0 THEN 0 ELSE value END AS value FROM {h_relation}"


def softmax(h_relation: str) -> str:
    return f"SELECT row_id, EXP(value) / SUM(EXP(value)) OVER () AS value FROM {h_relation}"


def insert_krone(con: duckdb.DuckDBPyConnection, table_a: str, table_b: str, x: np.ndarray) -> tuple[str, str]:
    a, b = calculate_kronecker(x, k=1)
    con.execute(f"CREATE OR REPLACE TABLE {table_a} (row_id INTEGER, value DOUBLE)")
    for i, val in enumerate(a[:, 0]):
        con.execute(f"INSERT INTO {table_a} VALUES ({i + 1}, {val})")
    con.execute(f"CREATE OR REPLACE TABLE {table_b} (row_id INTEGER, value DOUBLE)")
    for i, val in enumerate(b[:, 0]):
        con.execute(f"INSERT INTO {table_b} VALUES ({i + 1}, {val})")

    return table_a, table_b


def linear_krone(cols: int, table_z_a: str, table_z_b: str, table_w_a: str, table_w_b: str, b_relation: str) -> str:
    query = f"""WITH
A AS (
SELECT * FROM {table_z_a} z, {table_w_a} w WHERE z.row_id = w.row_id
),
B AS (
SELECT * FROM {table_z_b} z, {table_w_b} w WHERE z.row_id = w.row_id
),
c AS (
"""
    terms = []
    for i in range(cols):
        sum_product = kronecker_sum_product(["value", i], cols, 1)
        terms.append(f"SELECT {i + 1} AS row_id, {sum_product} AS value")
    query += "\nUNION ALL\n".join(terms)
    query += "\n)\n"
    query += f"SELECT c.row_id, c.value + b.column0 AS value FROM c, {b_relation} b WHERE c.row_id = b.row_id"
    return query


def run():
    start = time.time()
    # Load the input
    input = insert(con, "input", x)

    # Inference query
    # FC1
    h1 = execute(con, "h1", linear(100, input, "fc1_weight_4x100", "fc1_bias_100x1"))
    z1 = execute(con, "z1", relu(h1))

    # FC2
    h2 = execute(con, "h2", linear(50, z1, "fc2_weight_100x50", "fc2_bias_50x1"))
    z2 = execute(con, "z2", relu(h2))

    # FC3
    h3 = execute(con, "h3", linear(3, z2, "fc3_weight_50x3", "fc3_bias_3x1"))
    z3 = execute(con, "output", softmax(h3))

    end = time.time()
    elapsed = end - start
    print(f"Elapsed: {elapsed * 1000:.0f}ms")

    # Get the output
    output = con.execute(f"SELECT * FROM {z3}").fetchall()
    print("Output:", output)

    return elapsed, output


def run_krone():
    start = time.time()
    # Load the input
    input_a, input_b = insert_krone(con, "input_a", "input_b", x)

    # Inference query
    # FC1
    h1 = execute(con, "h1_kron",
                 linear_krone(100, input_a, input_b, "fc1_weight_4x100_a", "fc1_weight_4x100_b", "fc1_bias_100x1"))
    z1 = execute(con, "z1_kron", relu(h1))
    z1_values = con.execute(f"SELECT value FROM {z1}").fetchall()
    z1_values = np.array(z1_values).reshape(100, 1)
    z1_a, z1_b = insert_krone(con, "z1_a", "z1_b", z1_values)

    # FC2
    h2 = execute(con, "h2_kron",
                 linear_krone(50, z1_a, z1_b, "fc2_weight_100x50_a", "fc2_weight_100x50_b", "fc2_bias_50x1"))
    z2 = execute(con, "z2_kron", relu(h2))
    z2_values = con.execute(f"SELECT value FROM {z2}").fetchall()
    z2_values = np.array(z2_values).reshape(50, 1)
    z2_a, z2_b = insert_krone(con, "z2_a", "z2_b", z2_values)

    # FC3
    h3 = execute(con, "h3_kron", linear_krone(3, z2_a, z2_b, "fc3_weight_50x3_a", "fc3_weight_50x3_b", "fc3_bias_3x1"))
    z3 = execute(con, "output_kron", softmax(h3))

    end = time.time()
    elapsed = end - start
    print(f"Elapsed: {elapsed * 1000:.0f}ms")

    # Get the output
    output = con.execute(f"SELECT * FROM {z3}").fetchall()
    print("Output:", output)

    return elapsed, output


# Load the dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Scale the features for better training
scaler = StandardScaler()
X = scaler.fit_transform(X)

# first input and target
x = X[0]
y_0 = y[0]

con = duckdb.connect('lm.db', read_only=False)

print("Default:")
run()
print("Kronecker:")
run_krone()

con.close()
print("Done!")
