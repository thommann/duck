# Load the dataset
import duckdb
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Load the dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Scale the features for better training
scaler = StandardScaler()
X = scaler.fit_transform(X)

# first input and target
x = X[0]
y_0 = y[0]

con = duckdb.connect('lm.db', read_only=False)

# Load the input
con.execute(f"CREATE OR REPLACE TABLE input (row_id INTEGER, value DOUBLE)")
for i, val in enumerate(x):
    con.execute(f"INSERT INTO input VALUES ({i+1}, {val})")

# Inference query

# FC1

w1 = "WITH combo AS (" \
     "SELECT * " \
     "FROM input i, fc1_weight_4x100 w " \
     "WHERE i.row_id = w.row_id)\n"
for i in range(100):
    w1 += \
        f"SELECT {i + 1} AS row_id, SUM(c.value * c.column{i:02}) AS value " \
        f"FROM combo c " \
        f"UNION ALL\n"

w1 = w1.rstrip("UNION ALL\n")

# FC2

w2 = "WITH combo AS (" \
     "SELECT * " \
     "FROM z1 z, fc2_weight_100x50 w " \
     "WHERE z.row_id = w.row_id)\n"
for i in range(50):
    w2 += \
        f"SELECT {i + 1} AS row_id, SUM(c.value * c.column{i:02}) AS value " \
        f"FROM combo c " \
        f"UNION ALL\n"

w2 = w2.rstrip("UNION ALL\n")

# FC3

w3 = "WITH combo AS (" \
     "SELECT * " \
     "FROM z2 z, fc3_weight_50x3 w " \
     "WHERE z.row_id = w.row_id)\n"
for i in range(3):
    w3 += \
        f"SELECT {i + 1} AS row_id, SUM(c.value * c.column{i:01}) AS value " \
        f"FROM combo c " \
        f"UNION ALL\n"

w3 = w3.rstrip("UNION ALL\n")

# Bias

bias = "SELECT a.row_id, a.value + b.column0 AS value "

# ReLU

relu = "SELECT h.row_id, CASE WHEN h.value < 0 THEN 0 ELSE h.value END AS value "

# Softmax

softmax = "WITH Exponents AS (" \
          "SELECT h.row_id, EXP(h.value) AS value " \
          "FROM h3 h),\n" \
          "Denominator AS (" \
          "SELECT SUM(value) AS value " \
          "FROM Exponents)\n" \
          "SELECT e.row_id, e.value / d.value AS value " \
          "FROM Exponents e, Denominator d"

# Output
# FC1
a1 = "CREATE OR REPLACE TABLE a1 AS (" + w1 + ")"
con.execute(a1)
h1 = "CREATE OR REPLACE TABLE h1 AS (" + bias + "FROM a1 a, fc1_bias_100x1 b WHERE a.row_id = b.row_id)"
con.execute(h1)
z1 = "CREATE OR REPLACE TABLE z1 AS (" + relu + "FROM h1 h)"
con.execute(z1)

# FC2
a2 = "CREATE OR REPLACE TABLE a2 AS (" + w2 + ")"
con.execute(a2)
h2 = "CREATE OR REPLACE TABLE h2 AS (" + bias + "FROM a2 a, fc2_bias_50x1 b WHERE a.row_id = b.row_id)"
con.execute(h2)
z2 = "CREATE OR REPLACE TABLE z2 AS (" + relu + "FROM h2 h)"
con.execute(z2)

# FC3
a3 = "CREATE OR REPLACE TABLE a3 AS (" + w3 + ")"
con.execute(a3)
h3 = "CREATE OR REPLACE TABLE h3 AS (" + bias + "FROM a3 a, fc3_bias_3x1 b WHERE a.row_id = b.row_id)"
con.execute(h3)
output = "CREATE OR REPLACE TABLE output AS (" + softmax + ")"
con.execute(output)

con.close()
print("Done!")
