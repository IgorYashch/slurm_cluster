Чтобы провести эксперимент:
1) Запустить сервер с режимом логирования данных: `python3 -m server logging --log_file experiment/logfile.csv`
2) Запустить автоматическую постановку задач в очередь при помощи скрипта `submit_jobs.ipynb`
3) Присоединить информацию с временем постановки задачи иначалом ее выполнения, чтобы узнать целевую переменную waittime: `join_target.ipynb`
4) Обучить модель прогнозирования (например, xgboost) как в `train_model.ipynb`
5) Запустить сервер в режиме прогнозирования: `python3 -m server prediction --model_path experiment/model.pickle --mode xgboost`