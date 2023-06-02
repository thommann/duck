from autobench.params import image, name, rows_base_10, cc
from src.image_to_db import image_to_db

k = 2

col = 4
rows = rows_base_10

for row in rows:
    print(f"rows: {row:,}", flush=True)
    image_to_db(image, (row, col), name, k=k, compress_cols=cc)
