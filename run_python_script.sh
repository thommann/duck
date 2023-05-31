#!/bin/bash

source venv/bin/activate
name=${1%.py}
nohup python3 "$1" >& "$name".log &
echo $! > "$name".pid
