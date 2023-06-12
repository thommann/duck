# Load the dataset
import duckdb
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

# first input and target
x = X[0]
print(x)
y_0 = y[0]
print(y_0)

con = duckdb.connect('lm.db', read_only=False)

# Load the input
con.execute(f"DROP TABLE IF EXISTS input")
con.execute(f"CREATE TABLE input (row_id INTEGER, value DOUBLE)")
for i, val in enumerate(x):
    con.execute(f"INSERT INTO input VALUES ({i}, {val})")

inference_query = "WITH combo AS (SELECT * FROM input i, fc1_weight_4x100 w1 WHERE i.row_id = w1.column000)"
for i in range(100):
    inference_query += f"SELECT c.row_id, SUM(c.value * c.column{i+1:03}) AS value FROM combo c GROUP BY c.row_id UNION ALL "

inference_query = inference_query[:-11]

print(inference_query)

inference_query = f"WITH h1 AS ({inference_query}) SELECT * FROM h1, fc1_bias_100x1 b1 WHERE h1.row_id = b1.column0"

# create the output table
con.execute(f"CREATE TABLE h1 AS {inference_query}")

con.close()
print("Done!")
