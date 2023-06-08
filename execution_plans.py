# execute the original and the kronecker query and compare the execution plans
import duckdb

from queries import queries

col_indices = [0, 1]
nr_cols = 2
nr_rows = 1000000
rank_k = 2
max_rank = 10

original, kronecker = queries(col_indices, nr_cols, rank_k, max_rank)

print(original)
print()
print(kronecker)

plan_original = f"data/plans/original_{nr_rows}x{nr_cols}_rank{rank_k}_cols{len(col_indices)}.txt"
plan_kronecker = f"data/plans/kronecker_{nr_rows}x{nr_cols}_rank{rank_k}_cols{len(col_indices)}.txt"
config = """
PRAGMA enable_profiling='QUERY_TREE';
PRAGMA threads=48;
"""
db = f"data/databases/webb_{nr_rows}x{nr_cols}_rank_{max_rank}.db"

con = duckdb.connect(db)
con.execute(config)
con.execute(f"PRAGMA profile_output='{plan_original}';")
con.execute(original)
con.execute(f"PRAGMA profile_output='{plan_kronecker}';")
con.execute(kronecker)
con.close()
