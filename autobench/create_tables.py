from autobench.params import rows, cols, image, name
from src.image_to_db import image_to_db

for row in rows:
    for col in cols:
        print(f"row: {row}, col: {col}", flush=True)
        image_to_db(image, (row, col), name)
