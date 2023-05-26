import json

import duckdb

config = \
    "PRAGMA enable_profiling='json';" \
    "PRAGMA profile_output='data/profiles/webb.json';" \
    "EXPLAIN ANALYZE "

query1 = \
    "SELECT SUM(og.column0) FROM original og;"

query2 = \
    "SELECT (SUM(a.column0) * SUM(b.column0)) FROM matrix_a a, matrix_b b;"

queries = [("Original", query1), ("Kronecker", query2)]

results = []

con = duckdb.connect(database='data/databases/webb.db')

for name, query in queries:
    print()
    print(name)
    con.execute(config + query)
    con.sql(query).show()
    res = con.sql(query).fetchall()
    results.append(res[0][0])

    with open('data/profiles/webb.json', 'r') as f:
        profile = json.load(f)
        # print(json.dumps(profile, indent=4, sort_keys=True))
        print(profile["result"], profile["timing"])

con.close()

print(f"Results:\t{results}")
print(f"Error:\t{abs((results[0] - results[1]) / results[0])}")
