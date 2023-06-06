import argparse

from src.profiling import query_profiling, query_results


def bench_sc(name: str,
             dimensions: tuple[int, int],
             k: int,
             max_rank: int = 10,
             database: str | None = None) -> tuple[tuple, tuple, tuple, tuple]:
    """
    Compute the error and speedup of the Kronecker sum and sumproduct algorithms compared to the original algorithm.
    :param database:
    :param name:
    :param dimensions:
    :param k: Rank of the Kronecker approximation
    :param max_rank: Maximum rank of the Kronecker decomposition
    :return: Returns the error and speedup of the Kronecker sum and sumproduct algorithms.
    """
    rows, cols = dimensions
    full_name = f"{name}_{rows}x{cols}_sc"
    if database is None:
        database = f"data/databases/{full_name}_rank_{max_rank}.db"
    matrix_a, matrix_b, original = "A", "B", "C"

    column_format = "03d" if cols > 100 else "02d" if cols > 10 else "01d"
    column_format_ab = "03d" if cols * max_rank > 100 else "02d" if cols * max_rank > 10 else "01d"

    original_sum = f"""
    SELECT SUM(column{0:{column_format}}) AS result FROM {original};
    """

    original_sumproduct = f"""
    SELECT SUM(column{0:{column_format}} * column{1:{column_format}}) AS result FROM {original};
    """

    kronecker_sum = kronecker_sumproduct = "SELECT "
    for r in range(k):
        idx_0 = 0 * max_rank + r
        kronecker_sum += \
            f"((SELECT SUM(column{idx_0:{column_format_ab}}) FROM {matrix_a}) * " \
            f"(SELECT SUM(column{idx_0:{column_format_ab}}) FROM {matrix_b})) + "
        for r_prime in range(k):
            idx_1 = 1 * max_rank + r
            kronecker_sumproduct += \
                f"((SELECT SUM(column{idx_0:{column_format_ab}} * column{idx_1:{column_format_ab}}) " \
                f"FROM {matrix_a}) * " \
                f"(SELECT SUM(column{idx_0:{column_format_ab}} * column{idx_1:{column_format_ab}}) " \
                f"FROM {matrix_b})) + "

    kronecker_sum = kronecker_sum[:-2] + "AS result;"
    kronecker_sumproduct = kronecker_sumproduct[:-2] + "AS result;"

    sum_results = query_results(original_sum, kronecker_sum, database)
    sum_times = query_profiling(original_sum, kronecker_sum, database)
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
    bench_sc(args["name"], args["dimensions"], args["rank"], database=args["database"])
