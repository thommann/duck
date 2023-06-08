rows_base_2 = [2 ** x for x in range(10, 25)]
rows_base_10 = [10 ** x for x in range(3, 8)]
# concatenate and sort
rows = sorted(rows_base_2 + rows_base_10)
cols = [2 ** x for x in range(1, 5)]

max_k = 10
k = 1

cc = False
sc = False

permutations = 100
nr_factors = 2

epochs = 1
runs = 1

image = 'data/images/webb.png'
name = 'webb'
