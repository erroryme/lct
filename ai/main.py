"""
Заглушка AI сервиса для анализа КТГ данных
В будущем здесь будет настоящая нейронная сеть для анализа кардиотокографии
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Service - CTG Analysis",
    description="Заглушка для AI сервиса анализа кардиотокографии",
    version="1.0.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic модели
class CTGDataPoint(BaseModel):
    timestamp: datetime
    fhr_bpm: int  # Частота сердечных сокращений плода
    uterine_tone: float  # Тонус матки

class CTGAnalysisRequest(BaseModel):
    study_id: int
    patient_id: int
    data_points: List[CTGDataPoint]
    ga_weeks: Optional[int] = None  # Гестационный возраст в неделях

class CTGAnalysisResult(BaseModel):
    study_id: int
    analysis_timestamp: datetime
    risk_score: float  # Оценка риска от 0.0 до 1.0
    risk_level: str  # "low", "medium", "high"
    findings: List[str]
    recommendations: List[str]
    confidence: float  # Уверенность в анализе от 0.0 до 1.0

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: datetime

# WebSocket соединения
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket подключен. Всего соединений: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket отключен. Всего соединений: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Ошибка отправки сообщения: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections.copy():
            await self.send_personal_message(message, connection)

manager = ConnectionManager()

# Заглушка для анализа КТГ
def analyze_ctg_data(request: CTGAnalysisRequest) -> CTGAnalysisResult:
    """
    Заглушка для анализа КТГ данных
    В реальной реализации здесь будет нейронная сеть
    """
    logger.info(f"Анализ КТГ данных для исследования {request.study_id}")
    
    # Простая заглушка - генерируем случайные результаты
    import random
    
    # Анализируем базовые параметры
    fhr_values = [dp.fhr_bpm for dp in request.data_points]
    uterine_values = [dp.uterine_tone for dp in request.data_points]
    
    avg_fhr = sum(fhr_values) / len(fhr_values) if fhr_values else 120
    avg_uterine = sum(uterine_values) / len(uterine_values) if uterine_values else 0.0
    
    # Простая логика для определения риска
    risk_score = 0.0
    findings = []
    recommendations = []
    
    # Анализ ЧСС плода
    if avg_fhr < 110:
        risk_score += 0.3
        findings.append("Брадикардия плода")
        recommendations.append("Требуется дополнительное наблюдение")
    elif avg_fhr > 160:
        risk_score += 0.2
        findings.append("Тахикардия плода")
        recommendations.append("Рекомендуется консультация специалиста")
    else:
        findings.append("ЧСС плода в пределах нормы")
    
    # Анализ тонуса матки
    if avg_uterine > 50:
        risk_score += 0.2
        findings.append("Повышенный тонус матки")
        recommendations.append("Мониторинг сократительной активности")
    
    # Добавляем случайный фактор для демонстрации
    risk_score += random.uniform(0.0, 0.3)
    risk_score = min(risk_score, 1.0)
    
    # Определяем уровень риска
    if risk_score < 0.3:
        risk_level = "low"
        if not recommendations:
            recommendations.append("Продолжить плановое наблюдение")
    elif risk_score < 0.7:
        risk_level = "medium"
        recommendations.append("Увеличить частоту мониторинга")
    else:
        risk_level = "high"
        recommendations.append("Немедленная консультация врача")
    
    confidence = random.uniform(0.7, 0.95)  # Заглушка уверенности
    
    return CTGAnalysisResult(
        study_id=request.study_id,
        analysis_timestamp=datetime.now(),
        risk_score=round(risk_score, 3),
        risk_level=risk_level,
        findings=findings,
        recommendations=recommendations,
        confidence=round(confidence, 3)
    )

# REST API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка состояния сервиса"""
    return HealthResponse(
        status="healthy",
        service="ai-service",
        version="1.0.0",
        timestamp=datetime.now()
    )

@app.post("/api/analyze/ctg", response_model=CTGAnalysisResult)
async def analyze_ctg(request: CTGAnalysisRequest):
    """Анализ КТГ данных"""
    try:
        if not request.data_points:
            raise HTTPException(status_code=400, detail="Нет данных для анализа")
        
        result = analyze_ctg_data(request)
        logger.info(f"Анализ завершен для исследования {request.study_id}: {result.risk_level}")
        
        return result
        
    except Exception as e:
        logger.error(f"Ошибка анализа КТГ: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {str(e)}")

@app.get("/api/analyze/status/{study_id}")
async def get_analysis_status(study_id: int):
    """Получение статуса анализа (заглушка)"""
    return {
        "study_id": study_id,
        "status": "completed",
        "timestamp": datetime.now(),
        "message": "Анализ завершен"
    }

# WebSocket endpoint
@app.websocket("/ws/analysis")
async def websocket_analysis(websocket: WebSocket):
    """WebSocket для real-time анализа"""
    await manager.connect(websocket)
    try:
        while True:
            # Получаем данные от клиента
            data = await websocket.receive_text()
            try:
                # Парсим JSON данные
                analysis_request = json.loads(data)
                
                # Создаем объект запроса
                request = CTGAnalysisRequest(**analysis_request)
                
                # Выполняем анализ
                result = analyze_ctg_data(request)
                
                # Отправляем результат обратно
                await manager.send_personal_message(
                    result.json(), 
                    websocket
                )
                
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    json.dumps({"error": "Неверный JSON формат"}),
                    websocket
                )
            except Exception as e:
                await manager.send_personal_message(
                    json.dumps({"error": f"Ошибка анализа: {str(e)}"}),
                    websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket отключен")

# Заглушка для имитации real-time анализа
async def simulate_realtime_analysis():
    """Имитация real-time анализа для демонстрации"""
    while True:
        await asyncio.sleep(30)  # Каждые 30 секунд
        
        # Генерируем случайные данные для демонстрации
        mock_data = {
            "study_id": 1,
            "patient_id": 1,
            "data_points": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "fhr_bpm": 140 + (i * 2),
                    "uterine_tone": 10.0 + (i * 0.5)
                }
                for i in range(5)
            ],
            "ga_weeks": 32
        }
        
        # Отправляем всем подключенным клиентам
        if manager.active_connections:
            await manager.broadcast(json.dumps({
                "type": "realtime_analysis",
                "data": mock_data,
                "timestamp": datetime.now().isoformat()
            }))

# Запуск фоновой задачи
@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    logger.info("AI Service запущен")
    # Запускаем имитацию real-time анализа
    asyncio.create_task(simulate_realtime_analysis())

@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при остановке"""
    logger.info("AI Service остановлен")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
