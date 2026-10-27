#!/usr/bin/env bash
nnUNetv2_predict -i "$1" -o "$2" -d 001 -c 2d -f 1 --save_probabilities
