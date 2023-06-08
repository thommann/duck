# execute the original and the kronecker query and compare the execution plans
import duckdb

from src.queries import queries

col_indices = [0]
nr_cols = 2
nr_rows = 1000000
rank_k = 1
max_rank = 10
individual = False
output = "txt"  # "txt" or "json"

original, kronecker = queries(col_indices, nr_cols, rank_k, max_rank)

print(original)
print()
print(kronecker)

suffix = "_sc" if individual else ""
plan_original = f"data/plans/original_{nr_rows}x{nr_cols}_rank{rank_k}_factors{len(col_indices)}{suffix}.{output}"
plan_kronecker = f"data/plans/kronecker_{nr_rows}x{nr_cols}_rank{rank_k}_factors{len(col_indices)}{suffix}.{output}"
db = f"data/databases/webb_{nr_rows}x{nr_cols}{suffix}_rank_{max_rank}.db"

config = f"""
PRAGMA enable_profiling='{'QUERY_TREE' if output == 'txt' else 'JSON'}';
PRAGMA threads=48;
"""

con = duckdb.connect(db)
con.execute(config)
con.execute(f"PRAGMA profile_output='{plan_original}';")
con.execute(original)
con.execute(f"PRAGMA profile_output='{plan_kronecker}';")
con.execute(kronecker)
con.close()
