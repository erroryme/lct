# app/main.py
import asyncio
import logging
import tempfile
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from .database import AsyncSessionLocal
from .models import Patient, CTGStudy
from .csv_emulator import emulate_bpm_data, emulate_uc_data, initialize_csv_emulator
from .websocket_manager import manager
from .ctg_report import generate_ctg_report

# --- Pydantic модели ---

class PatientCreate(BaseModel):
    fullName: str

class PatientResponse(BaseModel):
    id: str
    fullName: str
    lastStudyType: Optional[str] = None
    lastStudyDate: Optional[datetime] = None
    monitoringStatus: str  # "stable" | "warning" | "critical"
    birthDate: Optional[str] = None
    medicalRecord: Optional[str] = None

class StudyResponse(BaseModel):
    id: str
    patientId: str
    patientName: str
    modality: str
    status: str  # "ready" | "processing" | "scheduled"
    performedAt: datetime
    findingsSummary: Optional[str] = None

# --- Lifespan ---

stop_event = asyncio.Event()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация CSV эмулятора
    initialize_csv_emulator()
    
    # Запуск фоновых задач эмуляции
    bpm_task = asyncio.create_task(emulate_bpm_data(stop_event, manager))
    uc_task = asyncio.create_task(emulate_uc_data(stop_event, manager))
    yield
    stop_event.set()
    await asyncio.gather(bpm_task, uc_task, return_exceptions=True)

app = FastAPI(lifespan=lifespan)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене замените на конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- REST API ---

@app.get("/api/patients/recent", response_model=List[PatientResponse])
async def get_recent_patients():
    async with AsyncSessionLocal() as session:
        # Получаем пациентов с последними исследованиями
        # Простой вариант: последние 5 пациентов
        patients = await session.execute(
            "SELECT p.patient_id, p.full_name, p.birth_date FROM patients p ORDER BY p.patient_id DESC LIMIT 5"
        )
        rows = patients.fetchall()
        result = []
        for row in rows:
            # Определяем статус мониторинга (заглушка)
            status = "stable"  # можно улучшить позже
            result.append(PatientResponse(
                id=f"pt-{row.patient_id}",
                fullName=row.full_name,
                monitoringStatus=status,
                birthDate=row.birth_date.isoformat() if row.birth_date else None
            ))
        return result

@app.get("/api/patients/search", response_model=List[PatientResponse])
async def search_patients(query: str = Query(..., min_length=1)):
    async with AsyncSessionLocal() as session:
        stmt = """
            SELECT p.patient_id, p.full_name, p.birth_date
            FROM patients p
            WHERE p.full_name ILIKE :q
            ORDER BY p.patient_id DESC
            LIMIT 20
        """
        patients = await session.execute(stmt, {"q": f"%{query}%"})
        rows = patients.fetchall()
        result = []
        for row in rows:
            result.append(PatientResponse(
                id=f"pt-{row.patient_id}",
                fullName=row.full_name,
                monitoringStatus="stable",
                birthDate=row.birth_date.isoformat() if row.birth_date else None,
                medicalRecord=f"MR-PT-{row.patient_id}"
            ))
        return result

@app.post("/api/patients", response_model=PatientResponse, status_code=201)
async def create_patient(data: PatientCreate):
    async with AsyncSessionLocal() as session:
        from sqlalchemy import text
        # Простая вставка
        res = await session.execute(
            text("INSERT INTO patients (full_name, birth_date) VALUES (:name, CURRENT_DATE) RETURNING patient_id"),
            {"name": data.fullName}
        )
        patient_id = res.scalar_one()
        await session.commit()
        return PatientResponse(
            id=f"pt-{patient_id}",
            fullName=data.fullName,
            monitoringStatus="stable"
        )

@app.get("/api/studies/recent", response_model=List[StudyResponse])
async def get_recent_studies():
    async with AsyncSessionLocal() as session:
        stmt = """
            SELECT s.study_id, s.patient_id, s.started_at, s.comments,
                   p.full_name
            FROM ctg_studies s
            JOIN patients p ON s.patient_id = p.patient_id
            ORDER BY s.started_at DESC
            LIMIT 10
        """
        studies = await session.execute(stmt)
        rows = studies.fetchall()
        result = []
        for row in rows:
            result.append(StudyResponse(
                id=f"st-{row.study_id}",
                patientId=f"pt-{row.patient_id}",
                patientName=row.full_name,
                modality="Кардиотокография (КТГ)",
                status="ready",
                performedAt=row.started_at,
                findingsSummary=row.comments or "Исследование завершено"
            ))
        return result

# --- WebSocket ---

@app.websocket("/ws/bpm")
async def ws_bpm(websocket: WebSocket):
    await manager.connect("bpm", websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect("bpm", websocket)

@app.websocket("/ws/uc")
async def ws_uc(websocket: WebSocket):
    await manager.connect("uc", websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect("uc", websocket)

@app.websocket("/ws/ai")
async def ws_ai(websocket: WebSocket):
    await manager.connect("ai", websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect("ai", websocket)

# --- Генерация справки ---

def get_db_config():
    """Получение конфигурации базы данных для ctg_report"""
    import os
    from .database import engine
    
    # Получаем URL из engine
    url = str(engine.url)
    # Конвертируем SQLAlchemy URL в psycopg2 формат
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql://")
    
    return {"dsn": url}

@app.get("/api/reports/ctg/{study_id}")
async def generate_ctg_report_endpoint(study_id: int):
    """Генерация PDF справки по исследованию КТГ"""
    pdf_path = None
    try:
        # Проверяем существование исследования
        async with AsyncSessionLocal() as session:
            study = await session.get(CTGStudy, study_id)
            if not study:
                raise HTTPException(status_code=404, detail=f"Исследование с ID {study_id} не найдено")
        
        # Создаем временный файл для PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            pdf_path = tmp_file.name
        
        # Генерируем PDF
        db_config = get_db_config()
        result_path = generate_ctg_report(db_config, study_id, pdf_path)
        
        # Возвращаем файл с автоматической очисткой
        return FileResponse(
            path=result_path,
            filename=f"ctg_report_{study_id}.pdf",
            media_type="application/pdf",
            background=lambda: os.unlink(result_path) if os.path.exists(result_path) else None
        )
        
    except HTTPException:
        # Перебрасываем HTTP исключения как есть
        raise
    except Exception as e:
        # Удаляем временный файл в случае ошибки
        if pdf_path and os.path.exists(pdf_path):
            os.unlink(pdf_path)
        raise HTTPException(status_code=500, detail=f"Ошибка генерации справки: {str(e)}")

# --- Health check ---

@app.get("/health")
async def health():
    return {"status": "ok"}
