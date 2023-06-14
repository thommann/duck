import duckdb
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


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
    SELECT *
    FROM {z_relation} z, {w_relation} w
    WHERE z.row_id = w.row_id
),
a AS (
"""
    terms = []
    for i in range(cols):
        terms.append(f"SELECT {i + 1} AS row_id, SUM(c.value * c.column{i:{len(str(int(cols - 1))):02d}d}) AS value FROM combo c")
    query += "\nUNION ALL\n".join(terms)
    query += "\n)\n"
    query += f"SELECT a.row_id, a.value + b.column0 AS value FROM a, {b_relation} b WHERE a.row_id = b.row_id"
    return query


def relu(h_relation: str) -> str:
    return f"SELECT row_id, CASE WHEN value < 0 THEN 0 ELSE value END AS value FROM {h_relation}"


def softmax(h_relation: str) -> str:
    return f"SELECT row_id, EXP(value) / SUM(EXP(value)) OVER () AS value FROM {h_relation}"


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
output = execute(con, "output", softmax(h3))

# Get the output
output = con.execute(f"SELECT * FROM {output}").fetchall()
print("Output:", output)

con.close()
print("Done!")
