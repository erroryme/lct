import json
import psycopg2
from psycopg2 import sql
import socket

def load_config(path="db_config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def check_network(host, port, timeout=3):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False

def healthcheck():
    config = load_config()

    print("=== Healthcheck базы данных ===")

    # --- Сетевой уровень ---
    if check_network(config["host"], config["port"]):
        print(f"Сетевое соединение доступно ({config['host']}:{config['port']})")
    else:
        print(f"Нет сетевого доступа к {config['host']}:{config['port']}")
        return  # дальше нет смысла проверять

    # --- Подключение к базе ---
    try:
        conn = psycopg2.connect(**config)
        print("Подключение к базе успешно")
    except Exception as e:
        print("База недоступна:", e)
        return

    try:
        with conn.cursor() as cur:
            # Проверка расширения TimescaleDB
            cur.execute("SELECT extname FROM pg_extension WHERE extname='timescaledb';")
            has_timescaledb = cur.fetchone() is not None

            if has_timescaledb:
                print("Расширение TimescaleDB установлено")
            else:
                print("Расширение TimescaleDB НЕ найдено")

            # список таблиц для проверки
            tables = ["patients", "ctg_studies", "ctg_samples"]

            all_ok = True
            for t in tables:
                try:
                    cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(t)))
                    count = cur.fetchone()[0]
                    print(f"Таблица {t}: {count} записей")
                except Exception as e:
                    all_ok = False
                    print(f"Таблица {t}: ошибка -> {e}")

    finally:
        conn.close()

    # Итог
    print("=== Итог ===")
    if all_ok and has_timescaledb:
        print("База работает корректно")
    else:
        print("Есть предупреждения или ошибки — проверьте логи выше")

if __name__ == "__main__":
    healthcheck()
