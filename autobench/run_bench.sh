#!/bin/bash

# Flag indicating if MKL environment should be sourced
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

path="autobench/bench_all.sh"
filename=$(basename -- "$path")
name="${filename%.*}"

# Kill process if it is already running
if [ -f out/"$name".pid ]; then
    kill "$(cat out/"$name".pid)"
fi

# Your command
nohup bash "$path" >& out/"$name".log &

echo $! > out/"$name".pid
