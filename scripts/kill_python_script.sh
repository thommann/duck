#!/bin/bash

filename=$(basename -- "$1")
name="${filename%.*}"

kill "$(cat out/"$name".pid)"
