import itertools

col_indices = [0, 1, 2]
rank_k = 2

terms = itertools.product(*[itertools.product([idx], range(rank_k)) for idx in col_indices])

print(list(terms))
