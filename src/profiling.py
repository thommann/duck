import json

import duckdb

config = """
PRAGMA enable_profiling='json';
PRAGMA profile_output='out/.temp';
PRAGMA threads=12;
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
    error = abs((original_result - kronecker_result) / original_result)
    print(f"Error: {error:.4%}", flush=True)

    original_time, kronecker_time = query_profiling(original_query, kronecker_query, database)
    speedup = original_time / kronecker_time
    print(f"Speedup: {speedup:.1f}x", flush=True)

    return error, speedup


def query_results(original_query: str, kronecker_query: str, database: str) -> tuple[float, float]:
    con = duckdb.connect(database=database)

    original_result = con.sql(original_query).fetchall()[0][0]
    kronecker_result = con.sql(kronecker_query).fetchall()[0][0]

    con.close()

    return original_result, kronecker_result


def query_profiling(original_query: str,
                    kronecker_query: str,
                    database: str,
                    runs: int = 10,
                    epochs: int = 10) -> tuple[float, float]:
    original_timings = []
    kronecker_timings = []

    queries = [(original_query, original_timings),
               (kronecker_query, kronecker_timings)]

    con = duckdb.connect(database=database)
    con.execute(config)

    for epoch in range(epochs):
        for query, timings in queries:
            for run in range(runs):
                print(f"EPOCH {epoch + 1}/{epochs} - RUN {run + 1}/{runs}", flush=True, end="\r")

                res = con.sql(query).explain('analyze')
                res_json = json.loads(res)
                timings.append(res_json["timing"])

                print(" " * 100, end="\r", flush=True)

    con.close()

    original_timings_truncated = original_timings[1:]
    kronecker_timings_truncated = kronecker_timings[1:]

    average_original_time = sum(original_timings_truncated) / len(original_timings_truncated)
    average_kronecker_time = sum(kronecker_timings_truncated) / len(kronecker_timings_truncated)

    return average_original_time, average_kronecker_time
