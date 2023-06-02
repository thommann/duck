from autobench.params import rows_base_10, name, cc
from src.bench_rank_k import bench_rank_k

k = 2

col = 4
rows = rows_base_10

print(f"rank: {k}, cols: {col}", flush=True)
for row in rows:
    print(f"rows: {row:,}", flush=True)
    database = f"data/databases/{name}_{row}x{col}_rank_2.db"
    bench_rank_k(name, (row, col), k, database, cc=cc)
