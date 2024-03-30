### Запуск для прогнозирования
```sbatch --comment="predict-time" ... script.sh```

### Сборка
 ```gcc --shared -I /home/slurm/slurm_simulator_tools/slurm_simulator  plugin.c  utils.c -lcurl -o job_submit_predict.so -fPIC```

### Сборка для симулятора
- Для тестирования на симуляторе использовать `plugin_for_simulator.c`. 
- Он печатает резутат работы плагина в файл `/home/slurm/workdir/logfile.log` в формате `<Job ID>: <Answer message>`
-  ```gcc --shared -I /home/slurm/slurm_simulator_tools/slurm_simulator  plugin_for_simulator.c  utils.c -lcurl -o job_submit_predict.so -fPIC```