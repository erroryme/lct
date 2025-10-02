FROM python:3.10-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование модели и кода
COPY main.py .
COPY scaler_unified.pkl .
COPY unified_mlp.pt .
COPY thresholds.json .

# Запуск
CMD ["python", "main.py"]