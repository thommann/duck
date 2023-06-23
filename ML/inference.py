import time

import duckdb
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from ML.calculate_kronecker import calculate_kronecker
from ML.params import middle_layer
from src.queries import kronecker_sum_product


def insert(con: duckdb.DuckDBPyConnection, table: str, x: np.ndarray) -> str:
    con.execute(f"CREATE OR REPLACE TABLE {table} (row_id INTEGER, value DOUBLE)")
    for i, val in enumerate(x):
        con.execute(f"INSERT INTO {table} VALUES ({i + 1}, {val})")
    return table


def insert_positional(con: duckdb.DuckDBPyConnection, table: str, x: np.ndarray) -> str:
    con.execute(f"CREATE OR REPLACE TABLE {table} (value DOUBLE)")
    for val in x:
        con.execute(f"INSERT INTO {table} VALUES ({val})")
    return table


def insert_alt(con: duckdb.DuckDBPyConnection, table: str, x: np.ndarray) -> str:
    con.execute(f"CREATE OR REPLACE TABLE {table} (row_id INTEGER, value DOUBLE)")
    for i, val in enumerate(x):
        con.execute(f"INSERT INTO {table} VALUES ({i + 1}, {val})")
    con.execute(f"INSERT INTO {table} VALUES (0, 1)")
    return table


def insert_alt_pos(con: duckdb.DuckDBPyConnection, table: str, x: np.ndarray) -> str:
    con.execute(f"CREATE OR REPLACE TABLE {table} (value DOUBLE)")
    for val in x:
        con.execute(f"INSERT INTO {table} VALUES ({val})")
    con.execute(f"INSERT INTO {table} VALUES (1)")
    return table


def insert_krone(con: duckdb.DuckDBPyConnection, table_a: str, table_b: str, x: np.ndarray) -> tuple[str, str]:
    a, b = calculate_kronecker(x, k=1)
    con.execute(f"CREATE OR REPLACE TABLE {table_a} (row_id INTEGER, value DOUBLE)")
    for i, val in enumerate(a[:, 0]):
        con.execute(f"INSERT INTO {table_a} VALUES ({i + 1}, {val})")
    con.execute(f"CREATE OR REPLACE TABLE {table_b} (row_id INTEGER, value DOUBLE)")
    for i, val in enumerate(b[:, 0]):
        con.execute(f"INSERT INTO {table_b} VALUES ({i + 1}, {val})")

    return table_a, table_b


def insert_krone_alt_pos(con: duckdb.DuckDBPyConnection, table_a: str, table_b: str, x: np.ndarray) -> tuple[str, str]:
    a, b = calculate_kronecker(x, k=1)
    con.execute(f"CREATE OR REPLACE TABLE {table_a} (value DOUBLE)")
    for val in a[:, 0]:
        con.execute(f"INSERT INTO {table_a} VALUES ({val})")
    con.execute(f"INSERT INTO {table_a} VALUES (1)")
    con.execute(f"CREATE OR REPLACE TABLE {table_b} (value DOUBLE)")
    for val in b[:, 0]:
        con.execute(f"INSERT INTO {table_b} VALUES ({val})")
    con.execute(f"INSERT INTO {table_b} VALUES (1)")

    return table_a, table_b


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


def linear_positional(cols: int, z_relation: str, w_relation: str, b_relation: str) -> str:
    query = f"""WITH
    combo AS (
    SELECT * FROM {z_relation} POSITIONAL JOIN {w_relation}
    ),
    a AS (
    """
    terms = []
    for i in range(cols):
        terms.append(f"SELECT SUM(c.value * c.column{i:{len(str(int(cols - 1))):02d}d}) AS value "
                     f"FROM combo c")
    query += "\nUNION ALL\n".join(terms)
    query += "\n)\n"
    query += f"SELECT value + column0 AS value FROM a POSITIONAL JOIN {b_relation}"
    return query


def linear_pivot_pos(cols: int, z_relation: str, w_relation: str, b_relation: str) -> str:
    query = f"""WITH
    combo AS (
    SELECT * FROM {z_relation} POSITIONAL JOIN {w_relation}
    ),
    A AS (
    SELECT """
    terms = []
    for i in range(cols):
        terms.append(f"SUM(c.value * c.column{i:{len(str(int(cols - 1))):02d}d})")
    query += " , ".join(terms)
    query += """
    FROM combo c
    ),
    aT AS (
    UNPIVOT A ON COLUMNS(*) INTO NAME row_id VALUE value
    )
    """

    query += f"SELECT value + column0 AS value FROM aT POSITIONAL JOIN {b_relation}"
    return query


def linear_alt(cols: int, z_relation: str, fc_relation: str) -> str:
    terms_a = []
    terms_h = []
    for i in range(cols):
        terms_a.append(
            f"SUM(Z.value * FC.column{i:{len(str(int(cols - 1))):02d}d}) AS column{i:{len(str(int(cols - 1))):02d}d}")
        terms_h.append(f"SELECT {i + 1} AS row_id, SUM(column{i:{len(str(int(cols - 1))):02d}d}) AS value FROM A")
    h_query = "\nUNION ALL\n".join(terms_h)
    query = f"""WITH A AS (
    SELECT {", ".join(terms_a)}
    FROM {z_relation} Z, {fc_relation} FC
    WHERE Z.row_id = FC.row_id
    GROUP BY FC.type
    )
    {h_query}
    """
    return query


def linear_alt_pivot(cols: int, z_relation: str, fc_relation: str) -> str:
    terms_a = []
    for i in range(cols):
        terms_a.append(
            f"SUM(Z.value * FC.column{i:{len(str(int(cols - 1))):02d}d}) AS '{i + 1}'")
    query = f"""WITH 
    A AS (
    SELECT {", ".join(terms_a)}
    FROM {z_relation} Z, {fc_relation} FC
    WHERE Z.row_id = FC.row_id
    ),
    H AS (
        UNPIVOT A ON COLUMNS(*) INTO NAME row_id VALUE value
    )
    SELECT CAST(row_id AS INTEGER) row_id, value FROM H
    """
    return query


def linear_alt_pivot_pos(cols: int, z_relation: str, fc_relation: str) -> str:
    terms_a = []
    for i in range(cols):
        terms_a.append(
            f"SUM(value * column{i:{len(str(int(cols - 1))):02d}d})")
    query = f"""WITH 
    A AS (
    SELECT {", ".join(terms_a)}
    FROM {z_relation} POSITIONAL JOIN {fc_relation}
    )
    UNPIVOT A ON COLUMNS(*) INTO NAME row_id VALUE value
    """
    return query


def linear_krone(cols: int, table_z_a: str, table_z_b: str, table_w_a: str, table_w_b: str, b_relation: str) -> str:
    terms = []
    for i in range(cols):
        sum_product = kronecker_sum_product(["value", i], cols, 1)
        terms.append(f"SELECT {i + 1} AS row_id, {sum_product} AS value")
    query = f"""WITH
    A AS (
    SELECT * FROM {table_z_a} z, {table_w_a} w WHERE z.row_id = w.row_id
    ),
    B AS (
    SELECT * FROM {table_z_b} z, {table_w_b} w WHERE z.row_id = w.row_id
    ),
    c AS (
    {" UNION ALL ".join(terms)}
    )
    SELECT c.row_id, c.value + b.column0 AS value 
    FROM c, {b_relation} b WHERE c.row_id = b.row_id
    """
    print(query)
    return query


def linear_krone_alt_pivot_pos(cols: int, table_z_a: str, table_z_b: str, table_fc_a: str, table_fc_b: str) -> str:
    terms = []
    for i in range(cols):
        sum_product = kronecker_sum_product(["value", i], cols, 1)
        terms.append(f"{sum_product}")
    query = f"""WITH
    A AS (
    SELECT * FROM {table_z_a} POSITIONAL JOIN {table_fc_a}
    ),
    B AS (
    SELECT * FROM {table_z_b} POSITIONAL JOIN {table_fc_b}
    ),
    c AS (
    SELECT {", ".join(terms)}
    )
    UNPIVOT c ON COLUMNS(*) INTO NAME row_id VALUE value
    """
    print(query)
    return query


def relu(h_relation: str) -> str:
    return f"SELECT row_id, CASE WHEN value < 0 THEN 0 ELSE value END AS value FROM {h_relation}"


def relu_positional(h_relation: str) -> str:
    return f"SELECT CASE WHEN value < 0 THEN 0 ELSE value END AS value FROM {h_relation}"


def softmax(h_relation: str) -> str:
    return f"SELECT row_id, EXP(value) / SUM(EXP(value)) OVER () AS value FROM {h_relation}"


def softmax_positional(h_relation: str) -> str:
    return f"SELECT EXP(value) / SUM(EXP(value)) OVER () AS value FROM {h_relation}"


def execute(con: duckdb.DuckDBPyConnection, table: str, query: str) -> str:
    query = f"CREATE OR REPLACE TABLE {table} AS (\n{query}\n)"
    con.execute(query)
    return table


def run_default(con: duckdb.DuckDBPyConnection,
                insert: callable, linear: callable, relu: callable, softmax: callable,
                suffix: str = "") -> str:
    # Load the input
    input = insert(con, f"input{suffix}", x)

    # Inference query
    # FC1
    h1 = execute(con, f"h1{suffix}",
                 linear(middle_layer[0], input, f"fc1_weight_4x{middle_layer[0]}", f"fc1_bias_{middle_layer[0]}x1"))
    z1 = execute(con, f"z1{suffix}", relu(h1))

    # FC2
    h2 = execute(con, f"h2{suffix}", linear(middle_layer[1], z1, f"fc2_weight_{middle_layer[0]}x{middle_layer[1]}",
                                            f"fc2_bias_{middle_layer[1]}x1"))
    z2 = execute(con, f"z2{suffix}", relu(h2))

    # FC3
    h3 = execute(con, f"h3{suffix}", linear(3, z2, f"fc3_weight_{middle_layer[1]}x3", f"fc3_bias_3x1"))
    z3 = execute(con, f"output{suffix}", softmax(h3))

    return z3


def run_alt(con: duckdb.DuckDBPyConnection,
            insert: callable, linear: callable, relu: callable, softmax: callable,
            suffix: str = "") -> str:
    # Load the input
    input = insert(con, f"input{suffix}", x)

    # Inference query
    # FC1
    h1 = execute(con, f"h1{suffix}",
                 linear(middle_layer[0], input, f"fc1_4x{middle_layer[0]}"))
    z1 = execute(con, f"z1{suffix}", relu(h1))

    # FC2
    h2 = execute(con, f"h2{suffix}", linear(middle_layer[1], z1, f"fc2_{middle_layer[0]}x{middle_layer[1]}"))
    z2 = execute(con, f"z2{suffix}", relu(h2))

    # FC3
    h3 = execute(con, f"h3{suffix}", linear(3, z2, f"fc3_{middle_layer[1]}x3"))
    z3 = execute(con, f"output{suffix}", softmax(h3))

    return z3


def run_row_idx(con) -> str:
    return run_default(con, insert, linear, relu, softmax)


def run_positional(con) -> str:
    return run_default(con, insert_positional, linear_positional, relu_positional, softmax_positional, suffix="_pos")


def run_pivot_pos(con) -> str:
    return run_default(con, insert_positional, linear_pivot_pos, relu_positional, softmax_positional,
                       suffix="_pivot_pos")


def run_alt_row_idx(con) -> str:
    return run_alt(con, insert_alt, linear_alt, relu, softmax, suffix="_alt")


def run_alt_pivot(con) -> str:
    return run_alt(con, insert_alt, linear_alt_pivot, relu, softmax, suffix="_alt_pivot")


def run_alt_pivot_pos(con) -> str:
    return run_alt(con, insert_alt_pos, linear_alt_pivot_pos, relu_positional, softmax_positional,
                   suffix="_alt_pivot_pos")


def run_krone(con):
    # Load the input
    input_a, input_b = insert_krone(con, "input_kron_a", "input_kron_b", x)

    # Inference query
    # FC1
    h1 = execute(con, "h1_kron",
                 linear_krone(middle_layer[0], input_a, input_b, f"fc1_weight_4x{middle_layer[0]}_a",
                              f"fc1_weight_4x{middle_layer[0]}_b", f"fc1_bias_{middle_layer[0]}x1"))
    z1 = execute(con, "z1_kron", relu(h1))
    z1_values = con.execute(f"SELECT value FROM {z1}").fetchall()
    z1_values = np.array(z1_values).reshape(middle_layer[0], 1)
    z1_a, z1_b = insert_krone(con, "z1_a", "z1_b", z1_values)

    # FC2
    h2 = execute(con, "h2_kron",
                 linear_krone(middle_layer[1], z1_a, z1_b, f"fc2_weight_{middle_layer[0]}x{middle_layer[1]}_a",
                              f"fc2_weight_{middle_layer[0]}x{middle_layer[1]}_b", f"fc2_bias_{middle_layer[1]}x1"))
    z2 = execute(con, "z2_kron", relu(h2))
    z2_values = con.execute(f"SELECT value FROM {z2}").fetchall()
    z2_values = np.array(z2_values).reshape(middle_layer[1], 1)
    z2_a, z2_b = insert_krone(con, "z2_a", "z2_b", z2_values)

    # FC3
    h3 = execute(con, "h3_kron",
                 linear_krone(3, z2_a, z2_b, f"fc3_weight_{middle_layer[1]}x3_a", f"fc3_weight_{middle_layer[1]}x3_b",
                              f"fc3_bias_3x1"))
    z3 = execute(con, "output_kron", softmax(h3))

    return z3


def run_krone_alt_pivot_pos(con):
    # Load the input
    input_a, input_b = insert_krone_alt_pos(con, "input_kron_alt_a", "input_kron_alt_b", x)

    # Inference query
    # FC1
    h1 = execute(con, "h1_kron_alt",
                 linear_krone_alt_pivot_pos(middle_layer[0], input_a, input_b,
                                            f"fc1_4x{middle_layer[0]}_a", f"fc1_4x{middle_layer[0]}_b"))
    z1 = execute(con, "z1_kron_alt", relu(h1))
    z1_values = con.execute(f"SELECT value FROM {z1}").fetchall()
    z1_values = np.array(z1_values).reshape(middle_layer[0], 1)
    z1_a, z1_b = insert_krone_alt_pos(con, "z1_a", "z1_b", z1_values)

    # FC2
    h2 = execute(con, "h2_kron_alt",
                 linear_krone_alt_pivot_pos(middle_layer[1], z1_a, z1_b,
                                            f"fc2_{middle_layer[0]}x{middle_layer[1]}_a",
                                            f"fc2_{middle_layer[0]}x{middle_layer[1]}_b"))
    z2 = execute(con, "z2_kron_alt", relu(h2))
    z2_values = con.execute(f"SELECT value FROM {z2}").fetchall()
    z2_values = np.array(z2_values).reshape(middle_layer[1], 1)
    z2_a, z2_b = insert_krone_alt_pos(con, "z2_a", "z2_b", z2_values)

    # FC3
    h3 = execute(con, "h3_kron_alt",
                 linear_krone_alt_pivot_pos(3, z2_a, z2_b,
                                            f"fc3_{middle_layer[1]}x3_a", f"fc3_{middle_layer[1]}x3_b"))
    z3 = execute(con, "output_kron_alt", softmax(h3))

    return z3


def time_run(runner: callable, title: str):
    con = duckdb.connect(f'data/ml{middle_layer[0]}x{middle_layer[1]}.db', read_only=False)
    con.execute(f"PRAGMA max_expression_depth={np.max(middle_layer) * 10};")
    _ = runner(con)
    print(title, flush=True)
    times = []
    for i in range(0):
        start = time.time()
        _ = runner(con)
        s = time.time() - start
        times.append(s)
    con.close()
    print(f"Average: {np.mean(times) * 1000:.0f}ms", flush=True)
    print()


if __name__ == "__main__":
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

    # time_run(run_row_idx, "Default")
    # time_run(run_positional, "Default (with positional)")
    # time_run(run_pivot_pos, "Default (with pivot and positional)")
    # time_run(run_alt_row_idx, "Alternative")
    # time_run(run_alt_pivot, "Alternative (with pivot)")
    # time_run(run_alt_pivot_pos, "Alternative (with pivot and positional)")
    time_run(run_krone, "Kronecker")
    time_run(run_krone_alt_pivot_pos, "Kronecker (with pivot and positional)")

    print("Done!")
