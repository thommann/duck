import argparse

from profiling import print_error_and_speedup


def bench(name: str, dimensions: tuple[int, int]):
    database = f"data/databases/{name}.db"
    original = f"{name}_{dimensions[0]}x{dimensions[1]}"
    matrix_a = f"{original}_a"
    matrix_b = f"{original}_b"

    column = "column0" if dimensions[1] > 10 else "column"

    original_sum = f"""
    SELECT SUM({column}0) AS result FROM {original};
    """

    kronecker_sum = f"""
    SELECT
    (SELECT SUM(column0) FROM {matrix_a}) * 
    (SELECT SUM(column0) FROM {matrix_b}) AS result;
    """

    original_sumproduct = f"""
    SELECT SUM({column}0 * {column}1) AS result FROM {original};
    """

    kronecker_sumproduct = f"""
    SELECT
    (SELECT SUM(column0 * column0) FROM {matrix_a}) * 
    (SELECT SUM(column0 * column1) FROM {matrix_b}) AS result;
    """

    print_error_and_speedup(original_sum, kronecker_sum, database)
    print_error_and_speedup(original_sumproduct, kronecker_sumproduct, database)


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Benchmark Webb")
    parser.add_argument("--name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--dimensions", type=int, nargs=2, required=True, help="Dimensions of the dataset")
    args = vars(parser.parse_args())

    # Name must be valid identifier
    if not args["name"].isidentifier():
        raise ValueError("Name must be a valid identifier")

    # Dimensions must be positive
    if args["dimensions"][0] <= 0 or args["dimensions"][1] <= 0:
        raise ValueError("Dimensions must be positive")

    return args


if __name__ == "__main__":
    args = parse_args()
    bench(args["name"], args["dimensions"])
