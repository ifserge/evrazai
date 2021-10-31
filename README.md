# evrazai

Для запуска интерфейса необходимо положить все csv файлы в папку data, а также запустить streamlit

```bash
pip3 install streamlit
```
Ну и все остальные библиотеки) 

```bash
cd ui
streamlit run app.py --server.address=127.0.0.1
```

Или гораздо проще поднять контейнер, для этого - все достаточно в корне evrazai выполнить:
```bash
docker build -t evrazappui:latest -f ./dockerfiles/DockerfileUI .
docker run -d -p 127.0.0.1:8501:8501 evrazappui:latest
```