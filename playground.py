from profiling import query_results, query_profiling

original_query = """
SELECT SUM(column0) AS result FROM original;
"""

kronecker_query = """
SELECT
(SELECT SUM(column0) FROM matrix_a) * 
(SELECT SUM(column0) FROM matrix_b) AS result;
"""

database = "data/databases/webb.db"
original_result, kronecker_result = query_results(original_query, kronecker_query, database)
print(f"Error: {abs((original_result - kronecker_result) / original_result):.2%}")

original_time, kronecker_time = query_profiling(original_query, kronecker_query, database)
print(f"Speedup: {original_time / kronecker_time:.1f}x")
