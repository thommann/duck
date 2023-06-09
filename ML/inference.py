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

con.close()
print("Done!")
