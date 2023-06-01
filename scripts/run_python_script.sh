#!/bin/bash

source venv/bin/activate
name=${1%.py}
nohup python3 "$1" >& out/"$name".log &
echo $! > out/"$name".pid
