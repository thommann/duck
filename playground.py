import json
import os
import time

import duckdb

config = \
    "PRAGMA enable_profiling='json';" \
    "PRAGMA profile_output='data/profiles/webb.json';" \
    "EXPLAIN ANALYZE "

original_query = \
    "SELECT SUM(column0) AS result FROM original;"

kronecker_query = \
    "SELECT " \
    "(SELECT SUM(column0) FROM matrix_a) * " \
    "(SELECT SUM(column0) FROM matrix_b) AS result;"

results = []

original_timings = []
kronecker_timings = []

queries = [("Original", original_query, original_timings),
           ("Kronecker", kronecker_query, kronecker_timings)]

# con = duckdb.connect(database='data/databases/webb.db')
# for name, query, _ in queries:
#     print()
#     print(name)
#     con.sql(query).show()
#     res = con.sql(query).fetchall()
#     results.append(res[0][0])
# con.close()
#
# print(f"Error:\t{abs((results[0] - results[1]) / results[0]):.2%}")


for _ in range(1):
    for _, query, timings in queries:
        con = duckdb.connect(database='data/databases/webb.db')
        con.execute(config + query)
        con.close()

        # sleep for 1 second to make sure the file is written
        time.sleep(1)

        with open('data/profiles/webb.json', 'r') as f:
            profile = json.load(f)
            timings.append(profile["timing"])

            # Cross product children
            cp_children = profile["children"][0]["children"][0]["children"]
            print(json.dumps(cp_children, indent=4))

        # Delete file
        os.remove('data/profiles/webb.json')

print("Original timings:", original_timings)
print("Kronecker timings:", kronecker_timings)

average_original_time = sum(original_timings) / len(original_timings)
average_kronecker_time = sum(kronecker_timings) / len(kronecker_timings)

print(f"Speedup:\t{average_original_time / average_kronecker_time:.2f}x")
