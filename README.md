# evrazai

[EVRAZ AI CHALLENGE 2021](https://hackathon.evraz.com/) Репозиторий команды DSTeam для трека о продувки стали.

## Техническая часть

Для установки зависимостей сделать ```poetry install```

#### Для запуска интерфейса необходимо положить все csv файлы в папку data, а также запустить streamlit


```bash
cd ui
streamlit run app.py --server.address=127.0.0.1
```
Или гораздо проще поднять контейнер, для этого достаточно в корне evrazai выполнить:
```bash
docker build -t evrazappui:latest -f ./dockerfiles/DockerfileUI .
docker run -d -p 127.0.0.1:8501:8501 evrazappui:latest
```
И открыть Интерфейс(http://127.0.0.1:8501/)

#### Для запуска решения, генерирующего сабмит
```bash
cd notebooks
poetry run solution.sh
```

## Описание решения

В решении использованы реккурентные сети для сбора композиции лома и для анализа управляющих действий (produv) и данных газовых датчиков (gas).

![Схема сетки](https://github.com/ifserge/evrazai/blob/main/schema.png)

