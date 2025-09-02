#!/bin/bash
# Script: find_maxN.sh
# Purpose: Binary search for the largest N that does not cause OOM when running ./wa1-task3 N

# initial search range (adjust upper bound if GPU memory is large)
LOW=1875006250          # 最小值：10万
HIGH=20000000000
BEST=0

while [ $LOW -le $HIGH ]; do
    MID=$(( (LOW + HIGH) / 2 ))
    echo ">>> Testing N=$MID"

    # Run the program and capture error
    ./wa1-task3 $MID > run.log 2>&1
    RET=$?

    if [ $RET -eq 0 ]; then
        # success, so we can go higher
        BEST=$MID
        LOW=$(( MID + 1 ))
        echo "OK (keep) -> move LOW up"
    else
        # failure (likely OOM)
        HIGH=$(( MID - 1 ))
        echo "OOM or error -> move HIGH down"
    fi
done

echo
echo "======================================"
echo "Largest safe N (no OOM): $BEST"
echo "======================================"

