rows_base_2 = [2 ** x for x in range(10, 25)]
rows_base_10 = [10 ** x for x in range(3, 8)]
# concatenate and sort
rows = sorted(rows_base_2 + rows_base_10)
cols = [2 ** x for x in range(1, 5)]

# rows = [2 ** 24]  # TODO: remove
# cols = [8]  # TODO: remove

max_k = 10
ks = [3]

col_decompositions = ['cc']  # options: cc, sc, nc

permutations = 10
factors = [3]

epochs = 1
runs = 1

image = 'data/images/webb.png'
name = 'webb'

seed = 42
