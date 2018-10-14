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

testDist() {
    dist_cap=$(echo $dist | tr a-z A-Z)
    num_todo=$(grep "TODO" tests/test_$dist.py | wc -l)

    blueEcho "::: $dist_cap TESTS"
    if (( $num_todo != 0 )) 
    then
        redEcho "$num_todo TODO(S) REMAINING"
    fi
    python3 -m unittest -q tests/test_$dist.py
    echo ""
}

if [[ $# -eq 0 ]]; then
    dists=( \
        "normal" "binomial" "cuniform" "beta" "t" "f" "gamma" "poisson" \
        "cauchy" "laplace" "duniform" "weibull" "negbin" \
        )
    for dist in ${dists[@]}
    do
        testDist $dist
    done
else
    for dist in "$@"
    do
        testDist $dist
    done
fi

