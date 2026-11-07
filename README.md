Python 3.12.14.

Запуск из папки проекта:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe -m streamlit run app.py
```

Если окружение .venv312 уже создано, достаточно последней команды.

st.cache_resource хранит один объект модели между перезапусками скрипта. st.cache_data хранит таблицы и вероятности, возвращая их копии, чтобы изменения данных не затрагивали общий кэш.

Без декоратора joblib.load заново читает модель с диска при каждом изменении ползунка или другого элемента интерфейса.
