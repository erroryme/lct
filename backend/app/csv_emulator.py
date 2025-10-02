import asyncio
import pandas as pd
import time
import logging
import json
import os
from datetime import datetime, timezone
from typing import Optional
import numpy as np
from collections import deque
from scipy.signal import butter, lfilter, medfilt

from .config import config

class OnlineUCFilter:
    def __init__(self, fs=1.0, med_window=31, cutoff=0.05, max_gap_seconds=10):
        self.fs = fs
        self.med_window = med_window if med_window % 2 == 1 else med_window + 1
        self.cutoff = cutoff
        self.max_gap_seconds = max_gap_seconds  # Максимальный допустимый разрыв в секундах

        # Буфер для хранения данных (время, значение)
        self.data_buffer = deque()

        # Параметры фильтра Баттерворта
        nyq = 0.5 * fs
        wn = cutoff / nyq
        self.b, self.a = butter(N=4, Wn=wn, btype='low')
        self.zi = None  # Начальное состояние фильтра

        # Для интерполяции
        self.last_processed_time = None
        self.interp_buffer = deque()
        self.last_valid_time = None

    def _reset_state(self):
        """Сброс внутреннего состояния фильтров и буферов."""
        self.data_buffer.clear()
        self.interp_buffer.clear()
        self.zi = None
        self.last_processed_time = None
        self.last_valid_time = None

    def _update_buffer(self, time_sec, value):
        if not (0 <= value <= 300):
            return  # Игнорируем значения вне диапазона

        # Если есть большой разрыв, сбрасываем состояние
        if self.last_valid_time is not None and (time_sec - self.last_valid_time) > self.max_gap_seconds:
            self._reset_state()

        self.data_buffer.append((time_sec, value))
        self.last_valid_time = time_sec
        

    def process(self, time_sec, value):
        self._update_buffer(time_sec, value)
        return self._process_buffer()

    def _process_buffer(self):
        if not self.data_buffer:
            return []
        
        sorted_data = sorted(self.data_buffer, key=lambda x: x[0])
        times = np.array([x[0] for x in sorted_data])
        values = np.array([x[1] for x in sorted_data])

        if self.last_processed_time is None:
            self.last_processed_time = int(np.floor(times[0]))

        max_time = int(np.floor(times[-1]))
        new_times = []
        new_values = []

        # Если уже "уперлись" в будущее — ничего не делаем
        if self.last_processed_time > max_time:
            return []

        while self.last_processed_time <= max_time:
            t_interp = self.last_processed_time

            if t_interp < times[0]:
                if len(times) == 1:
                    val = values[0]
                else:
                    slope = (values[1] - values[0]) / (times[1] - times[0])
                    val = values[0] + slope * (t_interp - times[0])
            elif t_interp > times[-1]:
                if len(times) == 1:
                    val = values[0]
                else:
                    slope = (values[-1] - values[-2]) / (times[-1] - times[-2])
                    val = values[-1] + slope * (t_interp - times[-1])
            else:
                val = np.interp(t_interp, times, values)

            new_times.append(t_interp)
            new_values.append(val)
            self.last_processed_time += 1

        if not new_times:
            return []

        # Медианный фильтр
        full_buffer = list(self.interp_buffer) + new_values
        if len(full_buffer) < self.med_window:
            y_med = new_values
        else:
            y_med_full = medfilt(full_buffer, kernel_size=self.med_window)
            y_med = y_med_full[len(self.interp_buffer):]

        self.interp_buffer.extend(new_values)
        while len(self.interp_buffer) > self.med_window:
            self.interp_buffer.popleft()

        # Баттерворт
        if self.zi is None:
            self.zi = np.zeros(max(len(self.b), len(self.a)) - 1)
        y_filtered, self.zi = lfilter(self.b, self.a, y_med, zi=self.zi)

        return [float(v) for v in y_filtered]


class StreamingFetalHRProcessor:
    def __init__(self,
                 hr_min=70,
                 hr_max=200,
                 max_rate_change=10.0,      # уд/мин/сек
                 median_window_sec=15,      # окно медианного фильтра (в секундах)
                 kalman_Q_scale=1e-3,
                 kalman_R=25.0,
                 max_gap_sec=5):
        self.hr_min = hr_min
        self.hr_max = hr_max
        self.max_rate_change = max_rate_change
        self.median_window = median_window_sec
        self.kalman_Q_scale = kalman_Q_scale
        self.kalman_R = kalman_R
        self.max_gap_sec = max_gap_sec
        
        # Буфер для хранения точек текущей и предыдущих секунд (для медианы и градиента)
        self.buffer = deque()  # элементы: (t, hr)

        # Состояние Калмана
        self.x_kalman = None  # [ЧСС, скорость]
        self.P_kalman = None

        # Последнее обработанное целое время (секунда)
        self.last_output_sec = None

    def _kalman_init(self, z0):
        """Инициализация Калмана"""
        self.x_kalman = np.array([z0, 0.0])
        self.P_kalman = np.eye(2) * 1000.0

    def _kalman_update(self, z, dt=1.0):
        """Один шаг Калмана с моделью постоянной скорости"""
        if self.x_kalman is None:
            self._kalman_init(z)
            return self.x_kalman[0]

        F = np.array([[1, dt], [0, 1]])
        H = np.array([[1, 0]])
        Q = self.kalman_Q_scale * np.array([[dt**3/3, dt**2/2],
                                            [dt**2/2, dt]])
        R = np.array([[self.kalman_R]])

        # Предсказание
        x_pred = F @ self.x_kalman
        P_pred = F @ self.P_kalman @ F.T + Q

        # Обновление
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        self.x_kalman = x_pred + (K * y).flatten()
        self.P_kalman = (np.eye(2) - K @ H) @ P_pred

        return self.x_kalman[0]

    def add_point(self, t_sec, hr_value):
        """
        Добавить новую точку.
        Возвращает (timestamp_sec, filtered_hr) для следующей готовой секунды,
        либо (None, None), если ещё нечего выдавать.
        Может возвращать заполненные пропуски (экстраполяция Калманом).
        """
        # 1. Фильтрация по физиологии (если точка есть)
        if hr_value is not None and not (self.hr_min <= hr_value <= self.hr_max):
            hr_value = None  # помечаем как недействительную

        # 2. Добавить в буфер (даже если hr_value=None — для временной привязки пропусков не нужно)
        if hr_value is not None:
            self.buffer.append((t_sec, hr_value))

        current_sec = int(np.floor(t_sec))
        if self.last_output_sec is None:
            # Инициализация: начинаем с первой возможной секунды
            if self.buffer:
                first_t = min(t for t, _ in self.buffer)
                self.last_output_sec = int(np.floor(first_t)) - 1
            else:
                self.last_output_sec = current_sec - 1

        next_sec = self.last_output_sec + 1

        # Если следующая секунда слишком далеко в будущем — ждём данных
        if next_sec > current_sec:
            return None, None

        # Проверяем, не превышен ли допустимый пропуск
        gap = current_sec - self.last_output_sec
        if gap > self.max_gap_sec + 1:
            # Слишком большой разрыв — сбрасываем Калман и ждём новых данных
            self.x_kalman = None
            self.P_kalman = None
            self.last_output_sec = current_sec - 1
            self._cleanup_buffer(current_sec)
            return None, None

        # === Обработка секунды `next_sec` ===
        window_start = next_sec - 0.5
        window_end = next_sec + 0.5
        points_in_window = [(t, hr) for t, hr in self.buffer if window_start <= t < window_end]

        if points_in_window:
            # Есть данные — обрабатываем как раньше
            hr_avg = np.mean([hr for _, hr in points_in_window])

            recent_points = [(t, hr) for t, hr in self.buffer if t >= next_sec - self.median_window]
            if len(recent_points) >= 3:
                hr_vals = np.array([hr for _, hr in recent_points])
                hr_for_kalman = hr_avg  # можно улучшить, но оставим
            else:
                hr_for_kalman = hr_avg

            hr_filtered = self._kalman_update(hr_for_kalman, dt=1.0)
        else:
            # Нет данных — экстраполяция Калманом (предсказание без обновления)
            if self.x_kalman is None:
                # Нет состояния — не можем экстраполировать
                self.last_output_sec = next_sec
                self._cleanup_buffer(next_sec)
                return None, None

            # Выполняем только предсказание (без обновления)
            dt = 1.0
            F = np.array([[1, dt], [0, 1]])
            Q = self.kalman_Q_scale * np.array([[dt**3/3, dt**2/2],
                                                [dt**2/2, dt]])
            self.x_kalman = F @ self.x_kalman
            self.P_kalman = F @ self.P_kalman @ F.T + Q
            hr_filtered = self.x_kalman[0]

            # Опционально: проверка на выход за физиологические границы после экстраполяции
            if not (self.hr_min <= hr_filtered <= self.hr_max):
                hr_filtered = np.clip(hr_filtered, self.hr_min, self.hr_max)

        # Обновляем состояние
        self.last_output_sec = next_sec
        self._cleanup_buffer(next_sec)

        return next_sec, float(hr_filtered)

    def _cleanup_buffer(self, up_to_sec):
        """Удаляем старые точки из буфера (оставляем запас для медианы)"""
        cutoff = up_to_sec - self.median_window - 1
        while self.buffer and self.buffer[0][0] < cutoff:
            self.buffer.popleft()

logger = logging.getLogger(__name__)

# Глобальные обработчики
bpm_processor = StreamingFetalHRProcessor(median_window_sec=3)  # Уменьшаем окно для быстрого тестирования
uc_processor = OnlineUCFilter(fs=1.0, med_window=5, cutoff=0.05)

# Данные CSV
bpm_data: Optional[pd.DataFrame] = None
uc_data: Optional[pd.DataFrame] = None
bpm_index = 0
uc_index = 0

# Счетчики для статистики
bpm_data_count = 0
uc_data_count = 0
bpm_error_count = 0
uc_error_count = 0

def load_csv_data():
    """Загрузка данных из CSV файлов"""
    global bpm_data, uc_data, bpm_index, uc_index
    
    try:
        # Загружаем BPM данные
        bpm_data = pd.read_csv(config.bpm_file_path)
        if 'time_sec' not in bpm_data.columns or 'value' not in bpm_data.columns:
            raise ValueError(f"CSV файл {config.csv_file_bpm} должен содержать колонки 'time_sec' и 'value'")
        bpm_data = bpm_data.sort_values('time_sec').reset_index(drop=True)
        bpm_index = 0
        logger.info(f"[CSV] Загружено {len(bpm_data)} записей BPM из {config.csv_file_bpm}")
        
        # Загружаем UC данные
        uc_data = pd.read_csv(config.uc_file_path)
        if 'time_sec' not in uc_data.columns or 'value' not in uc_data.columns:
            raise ValueError(f"CSV файл {config.csv_file_uc} должен содержать колонки 'time_sec' и 'value'")
        uc_data = uc_data.sort_values('time_sec').reset_index(drop=True)
        uc_index = 0
        logger.info(f"[CSV] Загружено {len(uc_data)} записей UC из {config.csv_file_uc}")
        
    except Exception as e:
        logger.error(f"[CSV] Ошибка загрузки CSV файлов: {e}")
        raise

# =============== BPM Эмуляция ===============

async def emulate_bpm_data(stop_event: asyncio.Event, websocket_manager=None):
    """Эмуляция отправки BPM данных (с автоматическим циклическим повтором)"""
    global bpm_data_count, bpm_error_count, bpm_index
    
    if bpm_data is None:
        logger.error("[BPM] Данные CSV не загружены")
        return

    logger.info(f"[BPM] Начинаем эмуляцию {len(bpm_data)} записей BPM")
    start_time = time.time()
    
    try:
        while not stop_event.is_set():
            current_elapsed = time.time() - start_time
            next_time = bpm_data.iloc[bpm_index]['time_sec']
            
            adjusted_time = next_time / config.playback_speed
            if current_elapsed >= adjusted_time:
                raw_value = float(bpm_data.iloc[bpm_index]['value'])
                sec_offset = int(current_elapsed)
                
                logger.info(f"[BPM] Эмуляция данных: raw={raw_value}, offset={sec_offset}с")
                bpm_data_count += 1

                result = bpm_processor.add_point(sec_offset, raw_value)
                if result and result[0]:
                    fhr_value = int(round(result[1]))
                    logger.info(f"[BPM] Успешно обработано: FHR={fhr_value} BPM")

                    if websocket_manager:
                        msg = {
                            "time_sec": int(time.time()),
                            "value": fhr_value
                        }
                        await websocket_manager.broadcast("bpm", json.dumps(msg, ensure_ascii=False))
                        logger.debug(f"[BPM] Отправлено через WebSocket: {msg}")
                    else:
                        logger.info(f"[BPM] Данные для отправки: {fhr_value} BPM")
                else:
                    logger.debug(f"[BPM] Данные не прошли фильтрацию: raw={raw_value}")

                bpm_index += 1
                
                # === ВСЕГДА перезапускаем с начала при достижении конца ===
                if bpm_index >= len(bpm_data):
                    bpm_index = 0
                    start_time = time.time()
                    logger.info("[BPM] Достигнут конец данных — перезапуск воспроизведения")

            else:
                await asyncio.sleep(0.01)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"[BPM] Критическая ошибка: {e}")
    finally:
        logger.info(f"[BPM] Эмуляция завершена. Статистика: обработано={bpm_data_count}, ошибок={bpm_error_count}")


# =============== UC Эмуляция ===============

async def emulate_uc_data(stop_event: asyncio.Event, websocket_manager=None):
    """Эмуляция отправки UC данных (с автоматическим циклическим повтором)"""
    global uc_data_count, uc_error_count, uc_index
    
    if uc_data is None:
        logger.error("[UC] Данные CSV не загружены")
        return

    logger.info(f"[UC] Начинаем эмуляцию {len(uc_data)} записей UC")
    start_time = time.time()
    
    try:
        while not stop_event.is_set():
            current_elapsed = time.time() - start_time
            next_time = uc_data.iloc[uc_index]['time_sec']
            
            adjusted_time = next_time / config.playback_speed
            if current_elapsed >= adjusted_time:
                raw_value = float(uc_data.iloc[uc_index]['value'])
                sec_offset = int(current_elapsed)
                
                logger.info(f"[UC] Эмуляция данных: raw={raw_value}, offset={sec_offset}с")
                uc_data_count += 1

                results = uc_processor.process(sec_offset, raw_value)
                if results:
                    logger.info(f"[UC] Получено {len(results)} результатов обработки")
                    for uc_value in results:
                        logger.info(f"[UC] Успешно обработано: UC={uc_value}")

                        if websocket_manager:
                            msg = {
                                "time_sec": int(time.time()),
                                "value": round(uc_value, 1)
                            }
                            await websocket_manager.broadcast("uc", json.dumps(msg, ensure_ascii=False))
                            logger.debug(f"[UC] Отправлено через WebSocket: {msg}")
                        else:
                            logger.info(f"[UC] Данные для отправки: {uc_value} UC")
                else:
                    logger.debug(f"[UC] Данные не прошли фильтрацию: raw={raw_value}")

                uc_index += 1
                
                # === ВСЕГДА перезапускаем с начала при достижении конца ===
                if uc_index >= len(uc_data):
                    uc_index = 0
                    start_time = time.time()
                    logger.info("[UC] Достигнут конец данных — перезапуск воспроизведения")

            else:
                await asyncio.sleep(0.01)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"[UC] Критическая ошибка: {e}")
    finally:
        logger.info(f"[UC] Эмуляция завершена. Статистика: обработано={uc_data_count}, ошибок={uc_error_count}")

# =============== Инициализация ===============

def initialize_csv_emulator():
    """Инициализация CSV эмулятора"""
    # Проверяем существование файлов
    if not config.validate_paths():
        raise FileNotFoundError("CSV файлы не найдены. Проверьте конфигурацию.")
    
    load_csv_data()
    logger.info("[CSV] CSV Emulator инициализирован")

# =============== Простая эмуляция без WebSocket ===============

async def run_simple_emulation(duration_seconds: int = 60):
    """Простая эмуляция без WebSocket для тестирования"""
    stop_event = asyncio.Event()
    
    # Инициализируем эмулятор
    initialize_csv_emulator()
    
    # Запускаем задачи
    bpm_task = asyncio.create_task(emulate_bpm_data(stop_event))
    uc_task = asyncio.create_task(emulate_uc_data(stop_event))
    
    logger.info(f"Запуск эмуляции на {duration_seconds} секунд...")
    
    try:
        # Ждем указанное время
        await asyncio.sleep(duration_seconds)
    except KeyboardInterrupt:
        logger.info("Эмуляция остановлена пользователем")
    finally:
        stop_event.set()
        await asyncio.gather(bpm_task, uc_task, return_exceptions=True)
        logger.info("Эмуляция завершена")

async def run_continuous_emulation():
    """Непрерывная эмуляция без WebSocket для тестирования"""
    stop_event = asyncio.Event()
    
    # Инициализируем эмулятор
    initialize_csv_emulator()
    
    # Запускаем задачи
    bpm_task = asyncio.create_task(emulate_bpm_data(stop_event))
    uc_task = asyncio.create_task(emulate_uc_data(stop_event))
    
    logger.info("Запуск непрерывной эмуляции...")
    
    try:
        # Ждем бесконечно (до прерывания)
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Эмуляция остановлена пользователем")
    finally:
        stop_event.set()
        await asyncio.gather(bpm_task, uc_task, return_exceptions=True)
        logger.info("Эмуляция завершена")
