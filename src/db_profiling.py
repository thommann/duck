import json

import duckdb
import numpy as np

config = """
PRAGMA enable_profiling='json';
PRAGMA profile_output='out/.temp';
PRAGMA threads=48;
"""


def print_error_and_speedup(original_query: str, kronecker_query: str, database: str) -> tuple[float, float]:
    """
    Calculates the relative error and the speedup of the kronecker query compared to the original query.
    :param original_query:
    :param kronecker_query:
    :param database:
    :return: returns the relative error and the speedup
    """
    original_result, kronecker_result = query_results(original_query, kronecker_query, database)
    abs_error = abs(original_result - kronecker_result)
    rel_error = abs_error / original_result
    print(f"Abs. Error: {abs_error:.0f}", flush=True)
    print(f"Rel. Error: {rel_error:.4%}", flush=True)

    original_time, kronecker_time = query_profiling(original_query, kronecker_query, database)
    speedup = original_time / kronecker_time
    print(f"Speedup: {speedup:.1f}x", flush=True)

    return rel_error, speedup


def query_results(original_query: str, kronecker_query: str, database: str) -> tuple[float, float]:
    con = duckdb.connect(database=database, read_only=True)

    original_result = con.sql(original_query).fetchall()[0][0]
    kronecker_result = con.sql(kronecker_query).fetchall()[0][0]

    con.close()

    return original_result, kronecker_result


def query_profiling(original_query: str,
                    kronecker_query: str,
                    database: str,
                    runs: int = 1,
                    epochs: int = 1) -> tuple[float, float]:
    original_timings = []
    kronecker_timings = []

    queries = [(original_query, original_timings),
               (kronecker_query, kronecker_timings)]

    con = duckdb.connect(database=database, read_only=True)
    con.execute(config)

    for query, _ in queries:
        # warmup
        con.execute(query)

    for epoch in range(epochs):
        for query, timings in queries:
            for run in range(runs):
                res = con.sql(query).explain('analyze')
                res_json = json.loads(res)
                timings.append(res_json["timing"])

    con.close()

    average_original_time = float(np.mean(original_timings))
    average_kronecker_time = float(np.mean(kronecker_timings))

    return average_original_time, average_kronecker_time
