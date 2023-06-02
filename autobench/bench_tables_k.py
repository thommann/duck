from autobench.params import rows_base_10, name, cc
from src.bench_rank_k import bench_rank_k

k = 2

col = 4
rows = rows_base_10

print(f"rank: {k}, cols: {col}, cc: {cc}", flush=True)
cc_suffix = f"_cc" if cc else ""
for row in rows:
    print(f"rows: {row:,}", flush=True)
    database = f"data/databases/{name}_{row}x{col}{cc_suffix}_rank_2.db"
    bench_rank_k(name, (row, col), k, database, cc=cc)
