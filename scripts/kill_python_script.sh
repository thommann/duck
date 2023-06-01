#!/bin/bash

name=${1%.py}
kill "$(cat out/"$name".pid)"
