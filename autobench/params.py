rows_base_2 = [2 ** x for x in range(10, 25)]
rows_base_10 = [10 ** x for x in range(3, 8)]
# concatenate and sort
rows = sorted(rows_base_2 + rows_base_10)
cols = [2 ** x for x in range(1, 5)]

max_k = 10
ks = [1, 2, 3]

cc = False
sc = False

col_decompositions = ['cc', 'sc', 'nc']

permutations = 100
factors = [1, 2, 3]

epochs = 2
runs = 2

image = 'data/images/webb.png'
name = 'webb'
