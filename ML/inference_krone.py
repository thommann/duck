import time

import duckdb
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from ML.calculate_kronecker import calculate_kronecker
from src.queries import kronecker_sum_product





def execute(con: duckdb.DuckDBPyConnection, table: str, query: str) -> str:
    query = f"CREATE OR REPLACE TABLE {table} AS (\n{query}\n)"
    print(query)
    con.execute(query)
    return table





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

start = time.time()
# Load the input
input_a, input_b = insert_krone(con, "input_a", "input_b", x)

# Inference query
# FC1
h1 = execute(con, "h1_kron", linear_krone(100, input_a, input_b, "fc1_weight_4x100_a", "fc1_weight_4x100_b", "fc1_bias_100x1"))
z1 = execute(con, "z1_kron", relu(h1))
z1_values = con.execute(f"SELECT value FROM {z1}").fetchall()
z1_values = np.array(z1_values).reshape(100, 1)
z1_a, z1_b = insert_krone(con, "z1_a", "z1_b", z1_values)

# FC2
h2 = execute(con, "h2_kron", linear_krone(50, z1_a, z1_b, "fc2_weight_100x50_a", "fc2_weight_100x50_b", "fc2_bias_50x1"))
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

con.close()
print("Done!")
