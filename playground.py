import json

import duckdb

config = \
    "PRAGMA enable_profiling='json';" \
    "PRAGMA profile_output='data/profiles/webb.json';" \
    "EXPLAIN ANALYZE "

query1 = \
    "SELECT SUM(column0) AS result FROM original;"

query2 = \
    "SELECT " \
    "(SELECT SUM(column0) FROM matrix_a) * " \
    "(SELECT SUM(column0) FROM matrix_b) AS result;"

queries = [("Original", query1), ("Kronecker", query2)]

results = []
timings = []


for name, query in queries:
    print()
    print(name)

    con = duckdb.connect(database='data/databases/webb.db')
    con.execute(config + query)
    con.sql(query).show()
    res = con.sql(query).fetchall()
    con.close()

    results.append(res[0][0])

    with open('data/profiles/webb.json', 'r') as f:
        profile = json.load(f)
        timings.append(profile["timing"])

print("Results:", results)
print("Timings:", timings)
print()
print(f"Speedup:\t{timings[0] / timings[1]:.2f}x")
print(f"Error:\t{abs((results[0] - results[1]) / results[0]):.2%}")
