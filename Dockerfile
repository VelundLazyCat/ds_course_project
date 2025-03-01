FROM python:3.10-slim

# Встановлення залежностей
WORKDIR /app
COPY pyproject.toml poetry.lock /app/
RUN pip install poetry && poetry install --no-dev

# Копіюємо файли проєкту
COPY app.py functions.py /app/
COPY accuracy_ml.pkl accuracy_nn.pkl model_log.pkl model_NN1.keras model_sgd.pkl model_tree.pkl stacking_model_ml.pkl values.pkl /app/

# Відкриваємо порт для Streamlit
EXPOSE 8501

# Команда для запуску застосунку
CMD ["poetry", "run", "streamlit", "run", "app.py"]
