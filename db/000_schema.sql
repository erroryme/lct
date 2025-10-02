CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Пациентки (простая карточка)
CREATE TABLE patients (
  patient_id   BIGSERIAL PRIMARY KEY,
  full_name    TEXT NOT NULL,
  birth_date   DATE NOT NULL
);

-- Исследования КТГ (шапка)
-- duration_sec — просто число секунд; считаем на бэке
CREATE TABLE ctg_studies (
  study_id         BIGSERIAL PRIMARY KEY,
  patient_id       BIGINT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
  started_at       TIMESTAMPTZ NOT NULL,
  duration_sec     INTEGER,          -- опционально, можно NULL, считаем на бэке
  ga_weeks         SMALLINT,         -- гестационный срок на момент исследования
  doctor_full_name TEXT,             -- без отдельной таблицы врачей
  comments         TEXT
);

-- Посекундные сэмплы в Timescale
-- sec — смещение в секундах от started_at (для удобной загрузки формата {time_sec: value})
-- ts  — реальное время точки (для Timescale), заполняется бэком: ts = started_at + sec * 1s
CREATE TABLE ctg_samples (
  ts           TIMESTAMPTZ NOT NULL,
  study_id     BIGINT      NOT NULL REFERENCES ctg_studies(study_id) ON DELETE CASCADE,
  sec          INTEGER     NOT NULL,
  fhr_bpm      SMALLINT,           -- ЧСС
  uterine_tone NUMERIC,            -- тонус
  PRIMARY KEY (study_id, ts)
);

-- Преобразуем в hypertable по времени
SELECT create_hypertable('ctg_samples', 'ts', create_default_indexes => TRUE);
-- (дефолтные индексы Timescale создаст сам)

-- Удобный дополнительный индекс по смещению секунд (необязателен, но лёгкий)
CREATE INDEX ON ctg_samples (study_id, sec);
