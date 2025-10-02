import asyncio
import pandas as pd
import time
import logging
import json
import os
from datetime import datetime, timezone
from typing import Optional

from .config import config

# Импорт ваших обработчиков
try:
    from .bpm_flow import StreamingFetalHRProcessor
    from .uc_flow import OnlineUCFilter
except ImportError as e:
    raise ImportError("Модули bpm_flow или uc_flow не найдены. Убедитесь, что они в PYTHONPATH.") from e

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
