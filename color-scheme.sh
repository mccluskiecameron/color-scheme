#!/bin/bash

if [ $# -lt 2 ]; then
    echo "usage: $0 inp-file out-file"
    exit 1
fi

./farbfeld/bin/jpegff < $1 |
    xxd -ps -c 8 |
    tail -n+3 | sort | uniq |
    python3 color-scheme.py | xxd -ps -r | 
    ./farbfeld/bin/ffjpeg > $2
