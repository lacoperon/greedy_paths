#!/bin/bash
rm timing.txt
for i in `seq 100 50 2000`
do
  python greedy_lattice_test.py $i 2 2 True
done
