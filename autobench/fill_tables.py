from src.kronecker_to_db import kronecker_to_db

rows = [2 ** x for x in range(10, 26 + 1)]
cols = [2 ** x for x in range(1, 4 + 1)]

image = 'data/images/webb.png'
name = 'webb'

for row in rows:
    for col in cols:
        print(f"row: {row}, col: {col}", flush=True)
        input = f"data/matrices/{name}_{row}x{col}.csv"
        input_a = input.replace(".csv", "_a.csv")
        input_b = input.replace(".csv", "_b.csv")
        database = f"data/databases/{name}_{row}x{col}.db"
        kronecker_to_db(input, input_a, input_b, database)
