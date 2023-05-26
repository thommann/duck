from profiling import print_error_and_speedup

name = "webb"
dimensions = 1_000_000, 100

database = f"data/databases/{name}.db"
original = f"{name}_{dimensions[0]}x{dimensions[1]}"
matrix_a = f"{original}_a"
matrix_b = f"{original}_b"

original_sum = f"""
SELECT SUM(column00) AS result FROM {original};
"""

kronecker_sum = f"""
SELECT
(SELECT SUM(column0) FROM {matrix_a}) * 
(SELECT SUM(column0) FROM {matrix_b}) AS result;
"""

original_sumproduct = f"""
SELECT SUM(column00 * column01) AS result FROM {original};
"""

kronecker_sumproduct = f"""
SELECT
(SELECT SUM(column0 * column0) FROM {matrix_a}) * 
(SELECT SUM(column0 * column1) FROM {matrix_b}) AS result;
"""

print_error_and_speedup(original_sum, kronecker_sum, database)
print_error_and_speedup(original_sumproduct, kronecker_sumproduct, database)
