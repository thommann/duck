#!/bin/bash

# Flags indicating if nice should be used and if MKL environment should be sourced
USE_NICE=0
USE_MKL=0

# Parse command line options
while getopts "nm" opt; do
  case ${opt} in
    n)
      USE_NICE=1
      ;;
    m)
      USE_MKL=1
      ;;
    \?)
      echo "Invalid option: -$OPTARG" 1>&2
      exit 1
      ;;
  esac
done

# Remove parsed options and args from $@ list
shift $((OPTIND -1))

# Decide which python environment to activate
if [ "$USE_MKL" -eq 1 ]; then
    source ../intel/oneapi/setvars.sh
    source ../intel/oneapi/intelpython/latest/bin/activate
else
    source venv/bin/activate
fi

filename=$(basename -- "$1")
name="${filename%.*}"

# Kill process if it is already running
if [ -f out/"$name".pid ]; then
    kill "$(cat out/"$name".pid)"
fi

# Set current working directory to python file's directory
export PYTHONPATH="$PWD"/

# Your command
if [ "$USE_NICE" -eq 1 ]; then
    nohup nice python3 "$1" >& out/"$name".log &
else
    nohup python3 "$1" >& out/"$name".log &
fi

echo $! > out/"$name".pid
