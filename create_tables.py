from image_to_db import image_to_db

rows = [2 ** x for x in range(1, 24 + 1)]
cols = [2 ** x for x in range(1, 6 + 1)]

image = 'data/images/webb.png'
name = 'webb'

for row in rows:
    for col in cols:
        print(f"row: {row}, col: {col}")
        image_to_db(image, (row, col), name)
