#!/bin/bash

gcc --shared -I /home/slurm/slurm_simulator_tools/slurm_simulator  /home/slurm/workdir/plugin/plugin_for_simulator.c  /home/slurm/workdir/plugin/utils.c -lcurl -o job_submit_predict.so -fPIC

# Move the shared library to desired location
# cp job_submit_log.so /home/slurm/slurm_simulator_tools/install/slurm_programs/lib/slurm/

# Append line to the slurm.conf.template file 
# (ONLY ONE TIME!!!)
# echo 'JobSubmitPlugins=job_submit/predict' >> /home/slurm/slurm_simulator_tools/install/slurm_conf/slurm.conf.template
