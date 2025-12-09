#!/bin/bash
PYTHON_EXEC=/opt/homebrew/anaconda3/envs/mmauto/bin/python

for i in {3..9}
do
    $PYTHON_EXEC test_mmkg.py --filename locomo_hard_$i
done