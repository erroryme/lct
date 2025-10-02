# Медицинский проект - КТГ мониторинг

Система мониторинга кардиотокографии (КТГ) с AI анализом данных.

## Архитектура

Проект состоит из трех основных сервисов:

- **backend** - FastAPI приложение для обработки данных КТГ
- **ai-service** - AI сервис для анализа данных (заглушка)
- **timescaledb** - База данных TimescaleDB для хранения временных рядов

## Запуск проекта

### Предварительные требования

- Docker и Docker Compose
- Порты 8000, 8080, 5432 должны быть свободны

### Запуск

```bash
# Клонирование и переход в директорию
cd /home/serv_admin/back/med

# Запуск всех сервисов
docker-compose up --build

# Запуск в фоновом режиме
docker-compose up -d --build
```

### Проверка работы

После запуска сервисы будут доступны по адресам:

- **Backend API**: http://localhost:8000
- **AI Service**: http://localhost:8080
- **TimescaleDB**: localhost:5432

### Health checks

```bash
# Проверка backend
curl http://localhost:8000/health

# Проверка AI сервиса
curl http://localhost:8080/health

# Проверка статуса контейнеров
docker-compose ps
```

## API Endpoints

### Backend (порт 8000)

- `GET /health` - Проверка состояния
- `GET /api/patients/recent` - Список недавних пациентов
- `GET /api/patients/search?query=...` - Поиск пациентов
- `POST /api/patients` - Создание нового пациента
- `GET /api/studies/recent` - Список недавних исследований
- `GET /api/reports/ctg/{study_id}` - Генерация PDF отчета
- `WebSocket /ws/bpm` - Real-time данные ЧСС
- `WebSocket /ws/uc` - Real-time данные тонуса матки
- `WebSocket /ws/ai` - Real-time AI анализ

### AI Service (порт 8080)

- `GET /health` - Проверка состояния
- `POST /api/analyze/ctg` - Анализ КТГ данных
- `GET /api/analyze/status/{study_id}` - Статус анализа
- `WebSocket /ws/analysis` - Real-time анализ

## Структура проекта

```
med/
├── backend/                 # FastAPI приложение
│   ├── app/
│   │   ├── main.py         # Основное приложение
│   │   ├── models.py       # SQLAlchemy модели
│   │   ├── database.py     # Конфигурация БД
│   │   └── ...
│   ├── Dockerfile
│   └── requirements.txt
├── ai/                     # AI сервис (заглушка)
│   ├── main.py            # FastAPI приложение AI
│   ├── Dockerfile
│   └── requirements.txt
├── db/
│   └── 000_schema.sql     # Схема базы данных
├── docker-compose.yml     # Конфигурация Docker
└── config.env            # Переменные окружения
```

## Особенности

### AI Service (заглушка)

AI сервис реализован как заглушка с базовой логикой анализа:

- Анализ ЧСС плода (брадикардия/тахикардия)
- Анализ тонуса матки
- Генерация рекомендаций
- WebSocket для real-time анализа

В будущем здесь будет интегрирована настоящая нейронная сеть.

### База данных

Используется TimescaleDB для эффективного хранения временных рядов КТГ данных:

- Таблица `patients` - информация о пациентах
- Таблица `ctg_studies` - исследования КТГ
- Таблица `ctg_samples` - временные ряды данных (гипертаблица)

### Мониторинг

Все сервисы имеют health checks и автоматический перезапуск при сбоях.

## Разработка

### Логи

```bash
# Просмотр логов всех сервисов
docker-compose logs

# Логи конкретного сервиса
docker-compose logs backend
docker-compose logs ai-service
docker-compose logs timescaledb
```

### Остановка

```bash
# Остановка сервисов
docker-compose down

# Остановка с удалением volumes
docker-compose down -v
```

### Пересборка

```bash
# Пересборка конкретного сервиса
docker-compose build backend
docker-compose build ai-service

# Пересборка всех сервисов
docker-compose build
```

## Безопасность

⚠️ **Важно**: В продакшене необходимо:

1. Изменить пароли базы данных
2. Настроить правильные CORS политики
3. Использовать HTTPS
4. Настроить файрвол
5. Регулярно обновлять зависимости

## Поддержка

При возникновении проблем проверьте:

1. Статус контейнеров: `docker-compose ps`
2. Логи сервисов: `docker-compose logs [service]`
3. Доступность портов: `netstat -tlnp | grep :8000`
4. Health checks: `curl http://localhost:8000/health`

## Модульные тесты

Для проверки: back-end отправить curl запрос
    curl -X http://<your-ip>/health