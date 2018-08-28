#!/bin/bash

echo ""
BLUE="\033[1;34m"
ENDCOL="\033[0m"

colEcho() {
    echo -e "${BLUE}${1}${ENDCOL}"
}

dists=( "normal" "binomial" )

for dist in ${dists[@]}
do
    dist_cap=$(echo $dist | tr a-z A-Z)
    colEcho ":::$dist_cap TESTS"
    python3 -m unittest -q tests/test_$dist.py
    echo ""
done
