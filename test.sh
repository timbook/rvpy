#!/bin/bash

echo ""
BLUE="\033[1;34m"
RED="\033[0;31m"
ENDCOL="\033[0m"

blueEcho() {
    echo -e "${BLUE}${1}${ENDCOL}"
}

redEcho() {
    echo -e "${RED}${1}${ENDCOL}"
}

dists=( "normal" "binomial" "cuniform" )

for dist in ${dists[@]}
do
    dist_cap=$(echo $dist | tr a-z A-Z)
    num_todo=$(grep "TODO" tests/test_$dist.py | wc -l)

    blueEcho "::: $dist_cap TESTS"
    python3 -m unittest -q tests/test_$dist.py
    redEcho "$num_todo TODO(S) REMAINING"
    echo ""
done
