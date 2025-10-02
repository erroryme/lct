# app/websocket_manager.py
from fastapi import WebSocket
from typing import Dict, List

class BroadcastManager:
    def __init__(self):
        self.connections: Dict[str, List[WebSocket]] = {
            "bpm": [],
            "uc": [],
            "ai": []
        }

    async def connect(self, topic: str, websocket: WebSocket):
        await websocket.accept()
        if topic in self.connections:
            self.connections[topic].append(websocket)
            print('con')

    def disconnect(self, topic: str, websocket: WebSocket):
        if topic in self.connections and websocket in self.connections[topic]:
            self.connections[topic].remove(websocket)

    async def broadcast(self, topic: str, message: str):
        if topic not in self.connections:
            return
        disconnected = []
        for ws in self.connections[topic]:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(topic, ws)

manager = BroadcastManager()