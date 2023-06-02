rows_base_2 = [2 ** x for x in range(10, 27)]
rows_base_10 = [10 ** x for x in range(3, 9)]
# concatenate and sort
rows = sorted(rows_base_2 + rows_base_10)
cols = [2 ** x for x in range(1, 5)]

max_k = 10
k = 2

cc = False

image = 'data/images/webb.png'
name = 'webb'
