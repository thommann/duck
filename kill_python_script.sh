#!/bin/bash

name=${1%.py}
kill "$(cat "$name".pid)"
