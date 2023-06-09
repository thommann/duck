import time

from autobench.params import rows, cols, image, name, max_k, col_decompositions
from src.image_to_db import image_to_db

for col_decomposition in col_decompositions:
    cc = col_decomposition == "cc"
    sc = col_decomposition == "sc"
    for row in rows:
        for col in cols:
            start = time.time()
            print(f"rows: {row:,}, cols: {col:,}", flush=True)
            image_to_db(image, (row, col), name, k=max_k, cc=cc, sc=sc)
            end = time.time()
            print(f"All done! ({int(end - start)}s)", flush=True)
