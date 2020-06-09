#!/usr/bin/env bash
set -e -x
cd $(dirname "$0")/examples

function luigi() {
    ../luigi-qif.py "$@"
}

for counter in BSat ApproxMC; do
    luigi --model-counter $counter --priors sum.priors --mpc sum.mpc static-leakage 
    luigi --model-counter $counter --priors sum.priors --mpc sum.mpc dynamic-leakage 3
done

for example in dijkstra aid-dist sum vickrey; do
    luigi --model-counter ApproxMC --dump-circuit "$example.dot" --priors "$example.priors" --mpc "$example.mpc" static-leakage
done

dot -Tpng "sum.dot" > "sum.png" 2>/dev/null

echo "ALL TESTS PASSED"

