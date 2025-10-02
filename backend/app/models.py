# app/models.py
from sqlalchemy import Column, BigInteger, Text, Date, Integer, SmallInteger, Numeric, TIMESTAMP, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class Patient(Base):
    __tablename__ = "patients"
    patient_id = Column(BigInteger, primary_key=True)
    full_name = Column(Text, nullable=False)
    birth_date = Column(Date, nullable=False)

class CTGStudy(Base):
    __tablename__ = "ctg_studies"
    study_id = Column(BigInteger, primary_key=True)
    patient_id = Column(BigInteger, ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=False)
    started_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    duration_sec = Column(Integer)
    ga_weeks = Column(Integer)
    doctor_full_name = Column(Text)
    comments = Column(Text)

class CTGSample(Base):
    __tablename__ = "ctg_samples"
    ts = Column(TIMESTAMP(timezone=True), primary_key=True)
    study_id = Column(BigInteger, ForeignKey("ctg_studies.study_id", ondelete="CASCADE"), primary_key=True)
    sec = Column(Integer, nullable=False)
    fhr_bpm = Column(SmallInteger)
    uterine_tone = Column(Numeric)