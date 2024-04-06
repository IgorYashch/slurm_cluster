### Запуск с опцией прогнозирования
```sbatch --comment="predict-time" ... script.sh```

### Сборка
 ```gcc --shared -I </path/to/slurm/code/directory>  plugin.c  utils.c -lcurl -o job_submit_predict.so -fPIC```
