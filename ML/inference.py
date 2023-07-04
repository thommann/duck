import time

import duckdb
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch

from ML.calculate_kronecker import do_decomposition
from ML.params import middle_layer, use_sigmoid, k, max_k, \
    iris_default_relations, iris_alt_relations, iris_krone_relations, mnist_default_relations, mnist_alt_relations, \
    mnist_krone_relations, mnist_shapes, iris_shapes
from src.queries import kronecker_sum_product, kronecker_sum_product_separated


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
    a, b = do_decomposition(x, k=k)

    cols_a = a.shape[1]
    cols = ", ".join([f"value{j:{len(str(int(cols_a - 1))):02d}d} DOUBLE" for j in range(k)])
    con.execute(f"CREATE OR REPLACE TABLE {table_a} (row_id INTEGER, {cols})")
    for i, val in enumerate(a):
        con.execute(f"INSERT INTO {table_a} VALUES ({i + 1}, {', '.join([str(v) for v in val])})")

    cols_b = b.shape[1]
    cols = ", ".join([f"value{j:{len(str(int(cols_b - 1))):02d}d} DOUBLE" for j in range(k)])
    con.execute(f"CREATE OR REPLACE TABLE {table_b} (row_id INTEGER, {cols})")
    for i, val in enumerate(b):
        con.execute(f"INSERT INTO {table_b} VALUES ({i + 1}, {', '.join([str(v) for v in val])})")

    return table_a, table_b


def insert_krone_pos(con: duckdb.DuckDBPyConnection, table_a: str, table_b: str, x: np.ndarray) -> tuple[str, str]:
    a, b = do_decomposition(x, k=k)

    cols_a = a.shape[1]
    cols = ", ".join([f"value{j:{len(str(int(cols_a - 1))):02d}d} DOUBLE" for j in range(k)])
    con.execute(f"CREATE OR REPLACE TABLE {table_a} ({cols})")
    for val in a:
        con.execute(f"INSERT INTO {table_a} VALUES ({', '.join([str(v) for v in val])})")

    cols_b = b.shape[1]
    cols = ", ".join([f"value{j:{len(str(int(cols_b - 1))):02d}d} DOUBLE" for j in range(k)])
    con.execute(f"CREATE OR REPLACE TABLE {table_b} ({cols})")
    for val in b:
        con.execute(f"INSERT INTO {table_b} VALUES ({', '.join([str(v) for v in val])})")

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
    terms = []
    for i in range(cols):
        terms.append(f"SUM(c.value * c.column{i:{len(str(int(cols - 1))):02d}d})")
    query = f"""WITH
    combo AS (
    SELECT * FROM {z_relation} POSITIONAL JOIN {w_relation}
    ),
    A AS (
    SELECT {' , '.join(terms)} FROM combo c
    ),
    aT AS (
    UNPIVOT A ON COLUMNS(*) INTO NAME row_id VALUE value
    )
    SELECT value + column0 AS value FROM aT POSITIONAL JOIN {b_relation}
    """
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
        sum_product = kronecker_sum_product([0, i], cols, 1, max_rank=max_k, rank_k=k, col_names=["value", f"column"],
                                            col_numbers=[(1, 1), (cols, 1)])
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
    return query


def linear_krone_pivot_pos(cols: int, table_z_a: str, table_z_b: str, table_w_a: str, table_w_b: str,
                           b_relation: str) -> str:
    terms_a: list[str] = []
    terms_b: list[str] = []
    for i in range(cols):
        sums_a, sums_b = kronecker_sum_product_separated([0, i], cols, 1,
                                                         max_rank=max_k, rank_k=k, col_names=["value", f"column"],
                                                         col_numbers=[(1, 1), (cols, 1)])
        # concat sums to terms
        terms_a += sums_a
        terms_b += sums_b
    query = f"""WITH
    A AS (
    SELECT * FROM {table_z_a} POSITIONAL JOIN {table_w_a}
    ),
    B AS (
    SELECT * FROM {table_z_b} POSITIONAL JOIN {table_w_b}
    ),
    A_sums AS (
    SELECT {', '.join(terms_a)} FROM A
    ),
    B_sums AS (
    SELECT {', '.join(terms_b)} FROM B
    ),
    C AS (
    FROM (UNPIVOT A_sums ON COLUMNS(*) INTO NAME row_id_a VALUE value) a
    POSITIONAL JOIN (UNPIVOT B_sums ON COLUMNS(*) INTO NAME row_id_b VALUE value) b
    SELECT FLOOR((ROW_NUMBER() OVER () - 1) / {k ** 2}) AS col_id, a.value * b.value AS value
    ),
    C_sums AS (
        SELECT SUM(value) AS value FROM C GROUP BY col_id ORDER BY col_id
    )
    SELECT value + b.column0 AS value
    FROM C_sums POSITIONAL JOIN {b_relation} b
    """
    return query


def linear_krone_bert(shape_a: tuple[int, int], shape_b: tuple[int, int], table_z: str, table_w_a: str, table_w_b: str,
                      b_relation: str) -> str:
    vx_terms = []
    for i in range(shape_a[1]):
        vx_terms.append(f"CASE WHEN col_id = {i} THEN value END AS value{i}")

    vxx_terms = []
    for i in range(shape_a[1]):
        vxx_terms.append(f"(SELECT value{i} FROM VX WHERE value{i} IS NOT NULL)")

    query = f"""WITH
    X AS (
        SELECT FLOOR((ROW_NUMBER() OVER () - 1) / {shape_a[1]}) AS col_id, value AS value FROM {table_z}
    ),
    VX AS (
        SELECT {', '.join(vx_terms)} FROM X
    ),
    VXX AS (
        SELECT * FROM {' POSITIONAL JOIN '.join(vxx_terms)}
    ),
    B AS (
        SELECT * FROM {table_w_b} POSITIONAL JOIN VXX
    ),
    """

    # k times matmul v(B v^-1(x) A^T)
    col_a_format = f"{len(str(int(shape_a[0] * max_k - 1))):02d}"
    col_b_format = f"{len(str(int(shape_b[0] * max_k - 1))):02d}"
    for r in range(k):
        rows_bxt = []
        for row in range(shape_a[1]):
            cols_bxt = []
            for col in range(shape_b[0]):
                cols_bxt.append(f"SUM(column{shape_b[0] * r + col:{col_b_format}} * value{row}) AS value{col}")

            rows_bxt.append(f"SELECT {', '.join(cols_bxt)} FROM B")

        cols_bxa_v = []
        for col in range(shape_a[0]):
            rows_bxa_v = []
            for row in range(shape_b[0]):
                rows_bxa_v.append(f"SUM(value{row} * column{shape_a[0] * r + col:{col_a_format}})")
            cols_bxa_v += rows_bxa_v

        query += f"""
        BXT{r} AS (
            {' UNION ALL '.join(rows_bxt)}
        ),
        A{r} AS (
            SELECT * FROM BXT{r} POSITIONAL JOIN {table_w_a}
        ),
        BXA{r} AS (
            SELECT {', '.join(cols_bxa_v)} FROM A{r}
        ),
        Z{r} AS (
            UNPIVOT BXA{r} ON COLUMNS(*) INTO NAME row_id VALUE value{r}
        ),"""

    # remove last comma
    query = query[:-1]

    # sum up all Zs
    z_col_terms = []
    for r in range(k):
        z_col_terms.append(f"value{r}")

    z_rel_terms = []
    for r in range(k):
        z_rel_terms.append(f"Z{r}")

    query += f"""
    SELECT {' + '.join(z_col_terms)} + b.column0 AS value
    FROM {' POSITIONAL JOIN '.join(z_rel_terms)} POSITIONAL JOIN {b_relation} b
    """
    return query


def relu(h_relation: str) -> str:
    return f"SELECT row_id, CASE WHEN value < 0 THEN 0 ELSE value END AS value FROM {h_relation}"


def sigmoid(h_relation: str) -> str:
    return f"SELECT row_id, 1 / (1 + EXP(-value)) AS value FROM {h_relation}"


def relu_positional(h_relation: str) -> str:
    return f"SELECT CASE WHEN value < 0 THEN 0 ELSE value END AS value FROM {h_relation}"


def sigmoid_positional(h_relation: str) -> str:
    return f"SELECT 1 / (1 + EXP(-value)) AS value FROM {h_relation}"


activation = sigmoid if use_sigmoid else relu
activation_positional = sigmoid_positional if use_sigmoid else relu_positional


def softmax(h_relation: str) -> str:
    return f"SELECT row_id, EXP(value) / SUM(EXP(value)) OVER () AS value FROM {h_relation}"


def softmax_positional(h_relation: str) -> str:
    return f"SELECT EXP(value) / SUM(EXP(value)) OVER () AS value FROM {h_relation}"


def execute(con: duckdb.DuckDBPyConnection, table: str, query: str) -> str:
    query = f"CREATE OR REPLACE TABLE {table} AS (\n{query}\n)"
    con.execute(query)
    return table


def run_default(con: duckdb.DuckDBPyConnection,
                insert: callable, linear: callable, activation: callable, softmax: callable, sample_x, model: str,
                suffix: str = "") -> str:
    relations = iris_default_relations if model == "iris" else mnist_default_relations if model == "mnist" else None
    if relations is None:
        raise ValueError(f"Unknown model {model}")

    output_size = 3 if model == "iris" else 10 if model == "mnist" else None

    # Load the input
    input = insert(con, f"input{suffix}", sample_x)

    # Inference query
    # FC1
    h1 = execute(con, f"h1{suffix}",
                 linear(middle_layer[0], input, relations[0][0], relations[0][1]))
    z1 = execute(con, f"z1{suffix}", activation(h1))

    # FC2
    h2 = execute(con, f"h2{suffix}", linear(middle_layer[1], z1, relations[1][0], relations[1][1]))
    z2 = execute(con, f"z2{suffix}", activation(h2))

    # FC3
    h3 = execute(con, f"h3{suffix}", linear(output_size, z2, relations[2][0], relations[2][1]))
    z3 = execute(con, f"output{suffix}", softmax(h3))

    return z3


def run_alt(con: duckdb.DuckDBPyConnection,
            insert: callable, linear: callable, activation: callable, softmax: callable, sample_x, model: str,
            suffix: str = "") -> str:
    relations = iris_alt_relations if model == "iris" else mnist_alt_relations if model == "mnist" else None
    if relations is None:
        raise ValueError(f"Unknown model {model}")

    output_size = 3 if model == "iris" else 10 if model == "mnist" else None

    # Load the input
    input = insert(con, f"input{suffix}", sample_x)

    # Inference query
    # FC1
    h1 = execute(con, f"h1{suffix}",
                 linear(middle_layer[0], input, relations[0]))
    z1 = execute(con, f"z1{suffix}", activation(h1))

    # FC2
    h2 = execute(con, f"h2{suffix}", linear(middle_layer[1], z1, relations[1]))
    z2 = execute(con, f"z2{suffix}", activation(h2))

    # FC3
    h3 = execute(con, f"h3{suffix}", linear(output_size, z2, relations[2]))
    z3 = execute(con, f"output{suffix}", softmax(h3))

    return z3


def run_row_idx(con, sample_x, model: str) -> str:
    return run_default(con, insert, linear, activation, softmax, sample_x, model)


def run_positional(con, sample_x, model: str) -> str:
    return run_default(con, insert_positional, linear_positional, activation_positional, softmax_positional, sample_x,
                       model, suffix="_pos")


def run_pivot_pos(con, sample_x, model: str) -> str:
    return run_default(con, insert_positional, linear_pivot_pos, activation_positional, softmax_positional, sample_x,
                       model, suffix="_pivot_pos")


def run_alt_row_idx(con, sample_x, model: str) -> str:
    return run_alt(con, insert_alt, linear_alt, activation, softmax, sample_x, model, suffix="_alt")


def run_alt_pivot(con, sample_x, model: str) -> str:
    return run_alt(con, insert_alt, linear_alt_pivot, activation, softmax, sample_x, model, suffix="_alt_pivot")


def run_alt_pivot_pos(con, sample_x, model: str) -> str:
    return run_alt(con, insert_alt_pos, linear_alt_pivot_pos, activation_positional, softmax_positional, sample_x,
                   model, suffix="_alt_pivot_pos")


def run_krone(con, insert: callable, linear: callable, activation: callable, softmax: callable, sample_x, model: str,
              suffix="") -> str:
    relations = iris_krone_relations if model == "iris" else mnist_krone_relations if model == "mnist" else None
    if relations is None:
        raise ValueError(f"Unknown model {model}")

    output_size = 3 if model == "iris" else 10 if model == "mnist" else None

    # Load the input
    input_a, input_b = insert(con, f"input_kron{suffix}_a", f"input_kron{suffix}_b", sample_x)

    # Inference query
    # FC1
    h1 = execute(con, f"h1_kron{suffix}",
                 linear(middle_layer[0], input_a, input_b, relations[0][0], relations[0][1], relations[0][2]))
    z1 = execute(con, f"z1_kron{suffix}", activation(h1))
    z1_values = con.execute(f"SELECT value FROM {z1}").fetchall()
    z1_values = np.array(z1_values).reshape(middle_layer[0], 1)
    z1_a, z1_b = insert(con, f"z1{suffix}_a", f"z1{suffix}_b", z1_values)

    # FC2
    h2 = execute(con, f"h2_kron{suffix}",
                 linear(middle_layer[1], z1_a, z1_b, relations[1][0], relations[1][1], relations[1][2]))
    z2 = execute(con, f"z2_kron{suffix}", activation(h2))
    z2_values = con.execute(f"SELECT value FROM {z2}").fetchall()
    z2_values = np.array(z2_values).reshape(middle_layer[1], 1)
    z2_a, z2_b = insert(con, f"z2{suffix}_a", f"z2{suffix}_b", z2_values)

    # FC3
    h3 = execute(con, f"h3_kron{suffix}",
                 linear(output_size, z2_a, z2_b, relations[2][0], relations[2][1], relations[2][2]))
    z3 = execute(con, f"output_kron{suffix}", softmax(h3))

    return z3


def run_krone_row_idx(con, sample_x, model: str) -> str:
    return run_krone(con, insert_krone, linear_krone, activation, softmax, sample_x, model)


def run_krone_pivot_pos(con, sample_x, model: str) -> str:
    return run_krone(con, insert_krone_pos, linear_krone_pivot_pos, activation_positional, softmax_positional, sample_x,
                     model, suffix="_pivot_pos")


def run_krone_bert(con, sample_x, model: str):
    relations = iris_krone_relations if model == "iris" else mnist_krone_relations if model == "mnist" else None
    if relations is None:
        raise ValueError(f"Unknown model {model}")

    shapes = iris_shapes if model == "iris" else mnist_shapes if model == "mnist" else None

    # Load the input
    input_x = insert_positional(con, f"input_krone_bert", sample_x)

    # Inference query
    # FC1
    h1 = execute(con, f"h1_krone_bert",
                 linear_krone_bert(shapes[0][0], shapes[0][1], input_x, relations[0][0], relations[0][1],
                                   relations[0][2]))
    z1 = execute(con, f"z1_krone_bert", activation_positional(h1))

    # FC2
    h2 = execute(con, f"h2_krone_bert",
                 linear_krone_bert(shapes[1][0], shapes[1][1], z1, relations[1][0], relations[1][1], relations[1][2]))
    z2 = execute(con, f"z2_krone_bert", activation_positional(h2))

    # FC3
    h3 = execute(con, f"h3_krone_bert",
                 linear_krone_bert(shapes[2][0], shapes[2][1], z2, relations[2][0], relations[2][1], relations[2][2]))
    z3 = execute(con, f"output_krone_bert", softmax_positional(h3))

    return z3


def time_run(con: duckdb.DuckDBPyConnection, runner: callable, title: str, sample_x, target_y, model: str):
    con.execute(f"PRAGMA max_expression_depth={np.max(middle_layer) * 10};")
    out_relation: str = runner(con, sample_x, model)
    print(title, flush=True)
    times = []
    for i in range(10):
        start = time.time()
        _ = runner(con, sample_x, model)
        s = time.time() - start
        times.append(s)

    res: list[tuple[float]] = con.execute(f"SELECT value FROM {out_relation}").fetchall()
    # flatten the list
    res_list = [item for sublist in res for item in sublist]
    # res_list[0] += 0.5
    category = np.argmax(res_list)
    print(f"Result: {res_list}", flush=True)
    print(f"Category/Target: {category}/{target_y}", flush=True)
    print(f"Average: {np.mean(times) * 1000:.0f}ms", flush=True)
    print()
    return category


def inference(dataset: Bunch, model: str):
    # Load the dataset
    X = dataset.data
    y = dataset.target
    y = y.astype(int)

    # Scale the features for better training
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Take first sample as test
    # X_test = X[:1]
    # y_test = y[:1]

    con = duckdb.connect(f'data/{model}_{middle_layer[0]}x{middle_layer[1]}.db', read_only=False)

    outputs = []
    for x, y_0 in zip(X_test, y_test):
        # time_run(run_row_idx, "Default")
        # time_run(run_positional, "Default (with positional join)")
        # time_run(run_pivot_pos, "Default (with pivot and positional join)")
        # time_run(run_alt_row_idx, "Alternative")
        # output = time_run(run_alt_pivot, "Alternative (with pivot)", x, y_0)
        # output = time_run(con, run_alt_pivot_pos, "Alternative (with pivot and positional join)", x, y_0, model)
        # output = time_run(run_krone_row_idx, "Kronecker", x, y_0)
        # output = time_run(run_krone_pivot_pos, "Kronecker alternative (with pivot and positional join)", x, y_0)
        output = time_run(con, run_krone_bert, "Kronecker BERT", x, y_0, model)
        outputs.append(output)
        time.sleep(5)

    con.close()

    # Print the accuracy
    print(f"Accuracy: {np.sum(np.array(outputs) == y_test) / len(y_test)}")

    print("Done!")


def inference_iris():
    iris = datasets.load_iris()
    inference(iris, "iris")


def inference_mnist():
    mnist = datasets.fetch_openml('mnist_784', version=1, cache=True, as_frame=False, parser='liac-arff')
    inference(mnist, "mnist")


if __name__ == "__main__":
    # inference_iris()
    inference_mnist()
