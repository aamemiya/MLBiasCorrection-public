#!/bin/bash -l 

count=0
[ -f log_sweep ] && rm log_sweep
rm log_sweep_temp_*
for val in 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.4 3.6 3.8 4.0 4.2 4.4;do
  count=$((count+1))
  python -u exp_sweep.py $val &> log_sweep_temp_$count &
done

wait
for i in `seq $count`;do
  tail -n 1  log_sweep_temp_$i >> log_sweep
done
rm log_sweep_temp_*

echo "done."
