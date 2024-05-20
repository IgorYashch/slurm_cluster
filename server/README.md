1. Запуск режима логирования: `python3 -m server logging --log_file experiment/logfile.csv`
2. Запуск режима прогнозирвоания: `python3 -m server prediction --model experiment/model.pickle --mode xgboost`
3. Остановить сервер можно командой 
    > kill -9 `lsof -i:4567 -t`