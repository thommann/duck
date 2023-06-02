rows_base_2 = [2 ** x for x in range(10, 26 + 1)]
# rows_base_10 = [10 ** x for x in range(3, 8 + 1)] TODO: add base 10
# concatenate and sort
# rows = sorted(rows_base_2 + rows_base_10) TODO: add base 10
rows = rows_base_2  # TODO: remove
cols = [2 ** x for x in range(1, 4 + 1)]

image = 'data/images/webb.png'
name = 'webb'
