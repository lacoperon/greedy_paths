#!/bin/bash
rm timing.txt
for i in `seq 100 50 2000`
do
  python greedy_lattice_ray_full.py $i 2 2 True
done
