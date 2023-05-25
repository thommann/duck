import sys
from io import StringIO

import duckdb

old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

con = duckdb.connect(database='data/databases/webb.db')
con.sql('EXPLAIN ANALYZE SELECT * FROM original').show()

sys.stdout = old_stdout

mystring = mystdout.getvalue()
mystring = mystring.replace('\n', '\n')

print(mystring)
