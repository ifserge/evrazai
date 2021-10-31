# evrazai

[EVRAZ AI CHALLENGE 2021](https://hackathon.evraz.com/) Репозиторий команды DSTeam для трека о продувки стали.

## Техническая часть

Для установки зависимостей сделать ```poetry install```

Для запуска интерфейса необходимо положить все csv файлы в папку data, а также запустить streamlit


```bash
cd ui
streamlit run app.py --server.address=127.0.0.1
```

Для запуска решения, генерирующего сабмит
```bash
cd notebooks
poetry run solution.sh
```

## Описание решения
