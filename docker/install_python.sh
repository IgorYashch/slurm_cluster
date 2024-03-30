#!/bin/bash

# Обновление системы и установка необходимых пакетов
    sudo yum update -y
    sudo yum install -y gcc openssl-devel bzip2-devel libffi-devel wget

# Скачивание исходного кода Python 3.8
sudo wget https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tgz

# Распаковка архива
sudo tar xzf Python-3.8.10.tgz

# Компиляция и установка Python 3.8
cd Python-3.8.10
sudo ./configure --enable-optimizations
sudo make altinstall

# Очистка загруженных файлов
sudo rm Python-3.8.10.tgz

# Вывод версии Python для проверки
alias python3=python3.8

