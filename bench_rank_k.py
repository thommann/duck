import argparse

from profiling import print_error_and_speedup


def bench_rank_k(name: str, dimensions: tuple[int, int], k: int) -> tuple[float, float, float, float]:
    """
    Compute the error and speedup of the Kronecker sum and sumproduct algorithms compared to the original algorithm.
    :param name:
    :param dimensions:
    :param k: Rank of the Kronecker approximation
    :return: Returns the error and speedup of the Kronecker sum and sumproduct algorithms.
    """
    full_name = f"{name}_{dimensions[0]}x{dimensions[1]}"
    database = f"data/databases/{full_name}_rank_{k}.db"
    original = full_name
    matrix_a = f"{original}_a"
    matrix_b = f"{original}_b"

    column = "column0" if dimensions[1] > 10 else "column"

    original_sum = f"""
    SELECT SUM({column}0) AS result FROM {original};
    """

    original_sumproduct = f"""
    SELECT SUM({column}0 * {column}1) AS result FROM {original};
    """

    k = 1

    kronecker_sum = kronecker_sumproduct = "SELECT "
    for rank in range(1, k + 1):
        kronecker_sum += \
            f"((SELECT SUM(column0) FROM {matrix_a}_{rank}) * (SELECT SUM(column0) FROM {matrix_b}_{rank})) + "
        for rank_prime in range(1, k + 1):
            kronecker_sumproduct += \
                f"(SELECT " \
                f"SUM((SELECT column0 FROM {matrix_a}_{rank}) * (SELECT column0 FROM {matrix_a}_{rank_prime})) * " \
                f"SUM((SELECT column0 FROM {matrix_b}_{rank}) * (SELECT column1 FROM {matrix_b}_{rank_prime}))) + "

    kronecker_sum = kronecker_sum[:-2] + "AS result;"
    kronecker_sumproduct = kronecker_sumproduct[:-2] + "AS result;"

    print(kronecker_sumproduct)

    sum_error, sum_speedup = print_error_and_speedup(original_sum, kronecker_sum, database)
    sumproduct_error, sumproduct_speedup = print_error_and_speedup(original_sumproduct, kronecker_sumproduct, database)

    return sum_error, sum_speedup, sumproduct_error, sumproduct_speedup


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Benchmark Webb")
    parser.add_argument("--name", "-n", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--dimensions", "-d", type=int, nargs=2, required=True, help="Dimensions of the dataset")
    parser.add_argument("--rank", "-r", "-k", type=int, required=True, help="Rank of the kronecker approximation")
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

    return args


if __name__ == "__main__":
    args = parse_args()
    bench_rank_k(args["name"], args["dimensions"], args["rank"])
