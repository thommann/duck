import duckdb
import torch

state_dict = torch.load("iris-model.pth")
print(state_dict.keys())

# load matrices into duckdb
con = duckdb.connect(database='lm.db', read_only=False)
for k, v in state_dict.items():
    k = k.replace(".", "_")
    if k.endswith("weight"):
        con.execute(f"CREATE TABLE {k} (row_id INTEGER, col_id INTEGER, value DOUBLE)")
        for i, row in enumerate(v):
            for j, val in enumerate(row):
                con.execute(f"INSERT INTO {k} VALUES ({i}, {j}, {val})")
    elif k.endswith("bias"):
        con.execute(f"CREATE TABLE {k} (row_id INTEGER, value DOUBLE)")
        for i, val in enumerate(v):
            con.execute(f"INSERT INTO {k} VALUES ({i}, {val})")
    else:
        raise Exception(f"Unknown key {k}")

con.close()
