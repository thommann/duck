import json
import time

import duckdb

config = """
PRAGMA enable_profiling='json';
PRAGMA profile_output='.temp';
PRAGMA threads=12;
"""


def print_error_and_speedup(original_query: str, kronecker_query: str, database: str):
    original_result, kronecker_result = query_results(original_query, kronecker_query, database)
    print(f"Error: {abs((original_result - kronecker_result) / original_result):.2%}")

    original_time, kronecker_time = query_profiling(original_query, kronecker_query, database)
    print(f"Speedup: {original_time / kronecker_time:.1f}x")


def query_results(original_query: str, kronecker_query: str, database: str) -> tuple[float, float]:
    con = duckdb.connect(database=database)
    con.execute(config)

    original_result = con.sql(original_query).fetchall()[0][0]
    kronecker_result = con.sql(kronecker_query).fetchall()[0][0]

    con.close()

    return original_result, kronecker_result


def query_profiling(original_query: str,
                    kronecker_query: str,
                    database: str,
                    runs: int = 10,
                    epochs: int = 1) -> tuple[float, float]:
    original_timings = []
    kronecker_timings = []

    queries = [(original_query, original_timings),
               (kronecker_query, kronecker_timings)]

    con = duckdb.connect(database=database)
    con.execute(config)

    for _ in range(epochs):
        for query, timings in queries:
            for _ in range(runs):
                res = con.sql(query).explain('analyze')

                res_json = json.loads(res)
                timings.append(res_json["timing"])

                # sleep for 1 second to make sure the file is written
                time.sleep(1)

    con.close()

    original_timings_truncated = original_timings[1:]
    kronecker_timings_truncated = kronecker_timings[1:]

    average_original_time = sum(original_timings_truncated) / len(original_timings_truncated)
    average_kronecker_time = sum(kronecker_timings_truncated) / len(kronecker_timings_truncated)

    return average_original_time, average_kronecker_time
