import time

from autobench.params import rows, cols, image, name, max_k, compress_cols, single_column
from src.image_to_db import image_to_db

assert not (compress_cols and single_column)

for row in rows:
    for col in cols:
        start = time.time()
        print(f"rows: {row:,}, cols: {col:,}", flush=True)
        image_to_db(image, (row, col), name, k=max_k, compress_cols=compress_cols, single_column=single_column)
        end = time.time()
        print(f"All done! ({int(end - start)}s)", flush=True)
