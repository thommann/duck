from autobench.params import rows, cols, image, name, max_k
from src.image_to_db import image_to_db

for row in rows:
    for col in cols:
        print(f"rows: {row:,}, cols: {col:,}", flush=True)
        image_to_db(image, (row, col), name, k=max_k)
