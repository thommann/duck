import argparse

from profiling import print_error_and_speedup


def bench_rank_k(name: str, dimensions: tuple[int, int], k: int, database: str | None = None) -> tuple[
    float, float, float, float]:
    """
    Compute the error and speedup of the Kronecker sum and sumproduct algorithms compared to the original algorithm.
    :param name:
    :param dimensions:
    :param k: Rank of the Kronecker approximation
    :return: Returns the error and speedup of the Kronecker sum and sumproduct algorithms.
    """
    full_name = f"{name}_{dimensions[0]}x{dimensions[1]}"
    if database is None:
        database = f"data/databases/{full_name}_rank_{k}.db"
    matrix_a, matrix_b, original = "A", "B", "C"

    column = "column0" if dimensions[1] > 10 else "column"

    original_sum = f"""
    SELECT SUM({column}0) AS result FROM {original};
    """

    original_sumproduct = f"""
    SELECT SUM({column}0 * {column}1) AS result FROM {original};
    """

    kronecker_sum = kronecker_sumproduct = "SELECT "
    for rank in range(1, k + 1):
        kronecker_sum += \
            f"((SELECT SUM(column0) FROM {matrix_a}_{rank}) * (SELECT SUM(column0) FROM {matrix_b}_{rank})) + "
        for rank_prime in range(1, k + 1):
            kronecker_sumproduct += \
                f"((SELECT SUM(a.column0 * a_prime.column0) FROM {matrix_a}_{rank} AS a JOIN {matrix_a}_{rank_prime} AS a_prime ON a.id = a_prime.id) * " \
                f"(SELECT SUM(b.column0 * b_prime.column1) FROM {matrix_b}_{rank} AS b JOIN {matrix_b}_{rank_prime} AS b_prime ON b.id = b_prime.id)) + "
    kronecker_sum = kronecker_sum[:-2] + "AS result;"
    kronecker_sumproduct = kronecker_sumproduct[:-2] + "AS result;"

    sum_error, sum_speedup = print_error_and_speedup(original_sum, kronecker_sum, database)
    sumproduct_error, sumproduct_speedup = print_error_and_speedup(original_sumproduct, kronecker_sumproduct, database)

    return sum_error, sum_speedup, sumproduct_error, sumproduct_speedup


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
    bench_rank_k(args["name"], args["dimensions"], args["rank"], database=args["database"])
