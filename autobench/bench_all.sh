#!/bin/bash

# Flags indicating if nice should be used and if MKL environment should be sourced
USE_MKL=0

# Parse command line options
while getopts "m" opt; do
  case ${opt} in
    m)
      USE_MKL=1
      ;;
    \?)
      echo "Invalid option: -$OPTARG" 1>&2
      exit 1
      ;;
  esac
done

# Decide which python environment to activate
if [ "$USE_MKL" -eq 1 ]; then
    source ../intel/oneapi/setvars.sh
    source ../intel/oneapi/intelpython/latest/bin/activate
else
    source venv/bin/activate
fi

# Set python path to current directory
export PYTHONPATH="$PWD"/

# Variants of the command to run
col_decompositions=("cc" "sc" "nc")
ranks=(1 2 3)
factors=(1 2 3)
for col_dec in "${col_decompositions[@]}"; do
  for k in "${ranks[@]}"; do
    for f in "${factors[@]}"; do
      python3 "autobench/bench_tables.py" -cd "$col_dec" -k "$k" -f "$f"
    done
  done
done
