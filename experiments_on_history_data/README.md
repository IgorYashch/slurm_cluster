В этой директории находится исходный код, необходимый для обработки двух датасетов (MIT и MUSTANG) и обучения на них классических моделей машинного обучения, а также нейросетевых моделей.

Как скачать данные:
1) Mustang: `wget -O ./data/mustang_release_v1.0beta.csv.gz https://ftp.pdl.cmu.edu/pub/datasets/ATLAS/mustang/mustang_release_v1.0beta.csv.gz`
2) MIT Supercloud скачивается отсюда: https://www.kaggle.com/datasets/skylarkphantom/mit-datacenter-challenge-data/data?select=scheduler_data.csv, затем его необходимо положить в `data/scheduler_data.csv`

Файлы в директории:
1) MIT_EDA.py / MUSTANDF_EDA.py - анализ и предлобработка датасетов
2) MIT_ML.py / MUSTANG_ML.py / MIT_xgboost.py - построение моделей машинного обучения на датасетах, полученных на предыдущем шаге
3) slurm_nn_data_preparation.py - файл содержит классы python с использованием модуля PyTorch и TorchLightning, необходимые для предобработки данных и создания формата, подходящего на вход нейросетям
4) slurm_nn_arch.py - файл содержит классы python с использованием модуля PyTorch и TorchLightning, описывающие архитектуру и параметризацию создаваемых нейросетей
5) MIT_MUSTANG_DL.py / MIT_MUSTANG_DL_testing.py - файлы для обучения и тестирования нейросетей, импортируют в себе файлы из пунктов 3 и 4