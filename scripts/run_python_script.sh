#!/bin/bash

# Flag indicating if nice should be used
USE_NICE=0

# Parse command line options
while getopts "n" opt; do
  case ${opt} in
    n)
      USE_NICE=1
      ;;
    \?)
      echo "Invalid option: -$OPTARG" 1>&2
      exit 1
      ;;
  esac
done

# Remove parsed options and args from $@ list
shift $((OPTIND -1))

source venv/bin/activate
filename=$(basename -- "$1")
name="${filename%.*}"

# Your command
if [ "$USE_NICE" -eq 1 ]; then
    nohup nice python3 "$1" >& out/"$name".log &
else
    nohup python3 "$1" >& out/"$name".log &
fi

echo $! > out/"$name".pid
