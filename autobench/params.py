rows_base_2 = [2 ** x for x in range(10, 27)]
rows_base_10 = [10 ** x for x in range(3, 9)]
# concatenate and sort
rows = sorted(rows_base_2 + rows_base_10)
cols = [2]

max_k = 10
k = 2

image = 'data/images/webb.png'
name = 'webb'
