#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#pip install psycopg2-binary reportlab

import json, os, sys
import psycopg2

# PDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ============================== Конфиг ==============================

def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_db_params(config):
    # Разрешаем либо DSN-строку, либо словарь параметров psycopg2
    if isinstance(config, str):
        return {"dsn": config}
    if isinstance(config, dict) and isinstance(config.get("dsn"), str) and config["dsn"].strip():
        return {"dsn": config["dsn"]}
    allowed = {"host","port","dbname","user","password","sslmode","connect_timeout","options"}
    return {k: v for k, v in (config or {}).items() if k in allowed}

def _pick_font():
    # Подключаем системный TTF с кириллицей
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            pdfmetrics.registerFont(TTFont("Base", p))
            return "Base"
    return "Helvetica"  # фолбэк; при отсутствии кириллического шрифта установите DejaVuSans/Arial

def _fmt(val, none="—"):
    return none if val is None else str(val)

# ============================== Данные (только из БД) ==============================

def _fetch_report(conn, study_id: int):
    with conn.cursor() as cur:
        # Шапка — без вычислений в коде; формат дат делает БД
        cur.execute("""
            SELECT
                p.full_name,
                to_char(p.birth_date, 'DD.MM.YYYY')              AS birth_date,
                s.study_id,
                to_char(s.started_at, 'DD.MM.YYYY HH24:MI')       AS started_at,
                s.duration_sec,                                    -- как есть (секунды)
                s.ga_weeks,
                s.doctor_full_name,
                s.comments
            FROM ctg_studies s
            JOIN patients p ON p.patient_id = s.patient_id
            WHERE s.study_id = %s
        """, (study_id,))
        row = cur.fetchone()
        if not row:
            raise SystemExit(f"Исследование study_id={study_id} не найдено")

        (full_name, birth_date, study_id, started_at, duration_sec,
         ga_weeks, doctor_full_name, comments) = row

        # Агрегаты — все считает БД, приводим к text на стороне БД
        cur.execute("""
            SELECT
                COUNT(*)::text                  AS n_all,
                COUNT(fhr_bpm)::text           AS n_fhr,
                COUNT(uterine_tone)::text      AS n_tone,
                AVG(fhr_bpm)::text             AS fhr_avg,
                MIN(fhr_bpm)::text             AS fhr_min,
                MAX(fhr_bpm)::text             AS fhr_max,
                AVG(uterine_tone)::text        AS tone_avg,
                MIN(uterine_tone)::text        AS tone_min,
                MAX(uterine_tone)::text        AS tone_max,
                MIN(sec)::text                 AS sec_min,
                MAX(sec)::text                 AS sec_max
            FROM ctg_samples
            WHERE study_id = %s
        """, (study_id,))
        stats = cur.fetchone()

    return {
        "full_name": full_name,
        "birth_date": birth_date,
        "study_id": study_id,
        "started_at": started_at,
        "duration_sec": duration_sec,          # без преобразований
        "ga_weeks": ga_weeks,
        "doctor_full_name": doctor_full_name,
        "comments": comments,
        # агрегаты из БД (уже text)
        "n_all": stats[0],
        "n_fhr": stats[1],
        "n_tone": stats[2],
        "fhr_avg": stats[3],
        "fhr_min": stats[4],
        "fhr_max": stats[5],
        "tone_avg": stats[6],
        "tone_min": stats[7],
        "tone_max": stats[8],
        "sec_min": stats[9],
        "sec_max": stats[10],
    }

# ============================== Рендер PDF ==============================

def _draw_line(c, x, y, label, value, font, size=11, leading=16):
    c.setFont(font, size)
    c.drawString(x, y, f"{label}: {value}")
    return y - leading

def _make_pdf(report: dict, out_path: str):
    font = _pick_font()
    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4
    margin = 36
    y = height - margin

    # Заголовок
    c.setFont(font, 16)
    c.drawCentredString(width/2, y, "Справка по исследованию КТГ")
    y -= 26

    # Шапка (значения как в БД)
    y = _draw_line(c, margin, y, "Пациент", _fmt(report["full_name"]), font, 12, 18)
    y = _draw_line(c, margin, y, "Дата рождения", _fmt(report["birth_date"]), font)

    y -= 6
    y = _draw_line(c, margin, y, "ID исследования", _fmt(report["study_id"]), font)
    y = _draw_line(c, margin, y, "Начало исследования", _fmt(report["started_at"]), font)
    y = _draw_line(c, margin, y, "Длительность (сек)", _fmt(report["duration_sec"]), font)
    y = _draw_line(c, margin, y, "Гестационный срок (нед)", _fmt(report["ga_weeks"]), font)
    y = _draw_line(c, margin, y, "Врач", _fmt(report["doctor_full_name"]), font)
    if report["comments"]:
        y = _draw_line(c, margin, y, "Комментарий", _fmt(report["comments"]), font)

    # Агрегаты (всё — текст из БД)
    y -= 10
    c.setFont(font, 12)
    c.drawString(margin, y, "Агрегаты (БД):")
    y -= 16
    c.setFont(font, 11)
    y = _draw_line(c, margin, y, "- Всего сэмплов", _fmt(report["n_all"]), font, 11, 14)
    y = _draw_line(c, margin, y, "- sec min", _fmt(report["sec_min"]), font, 11, 14)
    y = _draw_line(c, margin, y, "- sec max", _fmt(report["sec_max"]), font, 11, 14)
    y = _draw_line(c, margin, y, "- ЧСС avg", _fmt(report["fhr_avg"]), font, 11, 14)
    y = _draw_line(c, margin, y, "- ЧСС min", _fmt(report["fhr_min"]), font, 11, 14)
    y = _draw_line(c, margin, y, "- ЧСС max", _fmt(report["fhr_max"]), font, 11, 14)
    y = _draw_line(c, margin, y, "- Тонус avg", _fmt(report["tone_avg"]), font, 11, 14)
    y = _draw_line(c, margin, y, "- Тонус min", _fmt(report["tone_min"]), font, 11, 14)
    y = _draw_line(c, margin, y, "- Тонус max", _fmt(report["tone_max"]), font, 11, 14)

    c.showPage()
    c.save()

# ============================== Публичный API для бэка ==============================

def generate_ctg_report(db_config, study_id: int, out_path: str = "ctg_report.pdf"):
    """
    Использовать из бэкенда:
        - db_config: dict с полями psycopg2 (host/port/...) или DSN-строка
        - study_id:  int (ctg_studies.study_id)
        - out_path:  путь к PDF

    Возвращает путь к созданному PDF.
    """
    db_params = _load_db_params(db_config)
    conn = psycopg2.connect(db_params["dsn"]) if "dsn" in db_params else psycopg2.connect(**db_params)
    try:
        report = _fetch_report(conn, study_id)
    finally:
        conn.close()

    _make_pdf(report, out_path)
    return out_path

# ============================== Поведение при прямом запуске ==============================

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_cfg_path = os.path.join(base_dir, "db_config.json")
    run_cfg_path = os.path.join(base_dir, "run.json")

    if not os.path.exists(db_cfg_path):
        sys.stderr.write("Нет db_config.json рядом со скриптом.\n")
        sys.exit(1)

    db_config = _load_json(db_cfg_path)

    if not os.path.exists(run_cfg_path):
        sys.stderr.write("Нет run.json (ожидается хотя бы {\"study_id\": <id>}).\n")
        sys.exit(2)

    run_cfg = _load_json(run_cfg_path)
    study_id = run_cfg.get("study_id")
    if study_id is None:
        sys.stderr.write("В run.json отсутствует обязательный ключ \"study_id\".\n")
        sys.exit(3)

    out_path = run_cfg.get("out", "ctg_report.pdf")
    result = generate_ctg_report(db_config, int(study_id), out_path)
    print(f"Готово: {result}")


'''
Использование в бэке

from ctg_report_pdf import generate_ctg_report

# dict-конфиг или DSN-строка — как удобно вашему коду
db_config = {
    "host": "127.0.0.1",
    "port": 5432,
    "dbname": "ctg",
    "user": "postgres",
    "password": "postgres",
}

pdf_path = generate_ctg_report(db_config, study_id=42, out_path="/tmp/ctg_42.pdf")

'''