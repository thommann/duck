import sys
from time import sleep


def main():
    print("Hello World!", flush=True)
    sleep(10)
    # Print python version
    print(sys.version, flush=True)
    sleep(10)
    # Print python version info
    print(sys.version_info, flush=True)
    sleep(10)
    # Print python executable path
    print(sys.executable, flush=True)


if __name__ == "__main__":
    main()
