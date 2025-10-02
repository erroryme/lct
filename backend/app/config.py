import os
from typing import Optional

class CSVConfig:
    """Конфигурация для CSV reader"""
    
    def __init__(self):
        # Пути к CSV файлам
        self.csv_data_dir: str = os.getenv("CSV_DATA_DIR", "./app/test")
        self.csv_file_bpm: str = os.getenv("CSV_FILE_BPM", "20250901-01000001_1.csv")
        self.csv_file_uc: str = os.getenv("CSV_FILE_UC", "20250901-01000001_2.csv")
        
        # Настройки воспроизведения
        self.playback_speed: float = float(os.getenv("CSV_PLAYBACK_SPEED", "1.0"))
        self.loop_playback: bool = os.getenv("CSV_LOOP", "true").lower() == "true"  # По умолчанию включено
        
    @property
    def bpm_file_path(self) -> str:
        """Полный путь к файлу BPM данных"""
        return os.path.join(self.csv_data_dir, self.csv_file_bpm)
    
    @property
    def uc_file_path(self) -> str:
        """Полный путь к файлу UC данных"""
        return os.path.join(self.csv_data_dir, self.csv_file_uc)
    
    def validate_paths(self) -> bool:
        """Проверка существования CSV файлов"""
        bpm_exists = os.path.exists(self.bpm_file_path)
        uc_exists = os.path.exists(self.uc_file_path)
        
        if not bpm_exists:
            print(f"Ошибка: CSV файл BPM не найден: {self.bpm_file_path}")
        if not uc_exists:
            print(f"Ошибка: CSV файл UC не найден: {self.uc_file_path}")
            
        return bpm_exists and uc_exists

# Глобальный экземпляр конфигурации
config = CSVConfig()
