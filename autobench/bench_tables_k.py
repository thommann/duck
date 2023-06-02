from autobench.params import rows_base_10, name
from src.bench_rank_k import bench_rank_k

k = 2

col = 4
rows = rows_base_10

for row in rows:
    database = f"data/databases/{name}_{row}x{col}_rank_10.db"
    bench_rank_k(name, (row, col), k, database)
