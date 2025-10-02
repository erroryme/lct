import numpy as np
from scipy.signal import butter, lfilter, medfilt
from collections import deque

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