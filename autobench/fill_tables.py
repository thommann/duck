from autobench.params import rows, cols, name
from src.kronecker_to_db import kronecker_to_db

for row in rows:
    for col in cols:
        print(f"row: {row}, col: {col}", flush=True)
        input_mat = f"data/matrices/{name}_{row}x{col}.csv"
        input_a = input_mat.replace(".csv", "_a.csv")
        input_b = input_mat.replace(".csv", "_b.csv")
        database = f"data/databases/{name}_{row}x{col}.db"
        kronecker_to_db(input_mat, input_a, input_b, database)
