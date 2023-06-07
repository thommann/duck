import argparse

import numpy as np

from src.profiling import query_results, query_profiling


def col_indices(col_idx: int, b_cols: int) -> tuple[int, int]:
    col_idx_a = col_idx // b_cols
    col_idx_b = col_idx % b_cols
    return col_idx_a, col_idx_b


def bench(name: str,
          dimensions: tuple[int, int],
          k: int,
          max_rank: int = 10,
          database: str | None = None,
          cc: bool = False) -> tuple[tuple, tuple, tuple, tuple]:
    """
    Compute the error and speedup of the Kronecker sum and sumproduct algorithms compared to the original algorithm.
    :param database:
    :param name:
    :param dimensions:
    :param k: Rank of the Kronecker approximation
    :param max_rank: Rank of the kronecker decomposition
    :param cc: Whether to compress the columns
    :return: Returns the error and speedup of the Kronecker sum and sumproduct algorithms.
    """
    rows, cols = dimensions
    full_name = f"{name}_{rows}x{cols}"
    col_suffix = "_cc" if cc else ""
    if database is None:
        database = f"data/databases/{full_name}{col_suffix}_rank_{max_rank}.db"
    matrix_a, matrix_b, original = "A", "B", "C"

    column_format = "03d" if cols > 100 else "02d" if cols > 10 else "01d"

    original_sum = f"""
    SELECT SUM(column{0:{column_format}}) AS result FROM {original};
    """

    original_sumproduct = f"""
    SELECT SUM(column{0:{column_format}} * column{1:{column_format}}) AS result FROM {original};
    """

    if cc:
        # If the columns are compressed, we need to know how many columns there are in each kronecker matrix
        b_cols = np.loadtxt(f"data/bcols/{full_name}.csv", delimiter=",", dtype=int)
        a_cols = cols // b_cols
    else:
        # Otherwise, A will have the same number of columns as C and B will have exactly one column
        a_cols, b_cols = cols, 1

    # The column names are generated by DuckDB and depend on the number of columns in the kronecker matrices
    column_format_a = "03d" if a_cols * max_rank > 100 else "02d" if a_cols * max_rank > 10 else "01d"
    column_format_b = "03d" if b_cols * max_rank > 100 else "02d" if b_cols * max_rank > 10 else "01d"

    col_0_a, col_0_b = col_indices(0, b_cols)
    col_1_a, col_1_b = col_indices(1, b_cols)

    kronecker_sum = kronecker_sumproduct = "SELECT "
    for r in range(k):
        a_0_idx = r * a_cols + col_0_a
        b_0_idx = r * b_cols + col_0_b
        kronecker_sum += \
            f"((SELECT SUM(column{a_0_idx:{column_format_a}}) FROM {matrix_a}) * " \
            f"(SELECT SUM(column{b_0_idx:{column_format_b}}) FROM {matrix_b})) + "
        for r_prime in range(k):
            a_1_idx = r_prime * a_cols + col_1_a
            b_1_idx = r_prime * b_cols + col_1_b
            kronecker_sumproduct += \
                f"((SELECT SUM(column{a_0_idx:{column_format_a}} * column{a_1_idx:{column_format_a}}) " \
                f"FROM {matrix_a}) * " \
                f"(SELECT SUM(column{b_0_idx:{column_format_b}} * column{b_1_idx:{column_format_b}}) " \
                f"FROM {matrix_b})) + "

    kronecker_sum = kronecker_sum[:-2] + "AS result;"
    kronecker_sumproduct = kronecker_sumproduct[:-2] + "AS result;"

    print("SUM", flush=True)
    sum_results = query_results(original_sum, kronecker_sum, database)
    sum_times = query_profiling(original_sum, kronecker_sum, database)
    print("SUMPRODUCT", flush=True)
    sumproduct_results = query_results(original_sumproduct, kronecker_sumproduct, database)
    sumproduct_times = query_profiling(original_sumproduct, kronecker_sumproduct, database)

    return sum_results, sum_times, sumproduct_results, sumproduct_times


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Benchmark Webb")
    parser.add_argument("--name", "-n", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--dimensions", "-d", type=int, nargs=2, required=True, help="Dimensions of the dataset")
    parser.add_argument("--rank", "-r", "-k", type=int, required=True, help="Rank of the kronecker approximation")
    parser.add_argument("--database", "-db", type=str, required=False, help="Path to the database")
    args = vars(parser.parse_args())

    # Name must be valid identifier
    if not args["name"].isidentifier():
        raise ValueError("Name must be a valid identifier")

    # Dimensions must be positive
    if args["dimensions"][0] <= 0 or args["dimensions"][1] <= 0:
        raise ValueError("Dimensions must be positive")

    # Rank must be positive
    if args["rank"] <= 0:
        raise ValueError("Rank must be positive")

    # Database must be a db file
    if args["database"] is not None and not args["database"].endswith(".db"):
        raise ValueError("Database must be a db file")

    return args


if __name__ == "__main__":
    args = parse_args()
    bench(args["name"], args["dimensions"], args["rank"], database=args["database"])
