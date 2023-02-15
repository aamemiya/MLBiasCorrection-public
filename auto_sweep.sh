#!/bin/bash -l 

count=0
[ -f log_sweep ] && rm log_sweep
rm log_sweep_temp_*
for val in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0;do
  count=$((count+1))
  python -u exp_sweep.py $val &> log_sweep_temp_$count &
done

wait
for i in `seq $count`;do
  tail -n 1  log_sweep_temp_$i >> log_sweep
done
rm log_sweep_temp_*

echo "done."
