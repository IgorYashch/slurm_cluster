#!/bin/bash

gcc --shared -I /root/slurm/slurm-21.08.8 /root/plugin/plugin.c  /root/plugin/utils.c -lcurl -o job_submit_predict.so -fPIC

# Move the shared library to desired location
# cp job_submit_log.so /home/slurm/slurm_simulator_tools/install/slurm_programs/lib/slurm/