import numpy as np
from collections import deque
import pandas as pd


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



