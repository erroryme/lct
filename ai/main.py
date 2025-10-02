import asyncio
import json
import os
import numpy as np
import torch
import joblib
from websockets import connect
from collections import deque

# --- Настройки ---
BASE = os.path.dirname(__file__)
FASTAPI_WS_URL = os.getenv("FASTAPI_WS_URL", "ws://host.docker.internal:8000")
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "10"))

print(f"🔌 Подключение к бэкенду: {FASTAPI_WS_URL}")
print(f"🪟 Размер окна признаков: {WINDOW_SIZE}")

# --- Загрузка модели ---
from TEST_train_resnet_multihead import MultiHeadMLP, device

scaler = joblib.load(os.path.join(BASE, "scaler_unified.pkl"))
expected_dim = len(scaler.feature_names_in_)

if 2 * WINDOW_SIZE != expected_dim:
    raise ValueError(
        f"Несоответствие: WINDOW_SIZE={WINDOW_SIZE} → ожидается {2*WINDOW_SIZE} признаков, "
        f"но scaler обучен на {expected_dim}."
    )

model = MultiHeadMLP(in_dim=expected_dim).to(device)
model.load_state_dict(torch.load(os.path.join(BASE, "unified_mlp.pt"), map_location=device))
model.eval()

with open(os.path.join(BASE, "thresholds.json")) as f:
    thresholds = json.load(f)

print("✅ Модель загружена")

# --- Буферы ---
uc_values = deque(maxlen=WINDOW_SIZE)
bpm_values = deque(maxlen=WINDOW_SIZE)

# --- Слушатели ---
async def listen_uc():
    async with connect(f"{FASTAPI_WS_URL}/ws/uc") as ws:
        print("📡 Слушаю /ws/uc")
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            uc_values.append(float(data["value"]))

async def listen_bpm():
    async with connect(f"{FASTAPI_WS_URL}/ws/bpm") as ws:
        print("📡 Слушаю /ws/bpm")
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            bpm_values.append(float(data["value"]))

# --- Отправка в /ws/ai ---
async def send_to_ai():
    async with connect(f"{FASTAPI_WS_URL}/ws/ai") as ws:
        print("📤 Отправляю в /ws/ai")
        last_len_uc = 0
        last_len_bpm = 0
        while True:
            if len(uc_values) == WINDOW_SIZE and len(bpm_values) == WINDOW_SIZE:
                if len(uc_values) == last_len_uc and len(bpm_values) == last_len_bpm:
                    await asyncio.sleep(0.1)
                    continue

                try:
                    # ВАЖНО: порядок должен совпадать с обучением!
                    x_raw = np.concatenate([np.array(uc_values), np.array(bpm_values)]).astype(np.float32)
                    x_scaled = scaler.transform(x_raw.reshape(1, -1))
                    xb = torch.tensor(x_scaled).to(device)

                    with torch.no_grad():
                        s_logit, long_out = model(xb)
                        s_prob = torch.sigmoid(s_logit).cpu().item()
                        long_probs = torch.sigmoid(long_out).cpu().numpy().ravel()

                    status = "Норма"
                    if s_prob > thresholds["t_high"]:
                        status = "Патология"
                    elif s_prob > thresholds["t_mid"]:
                        status = "Подозрение"

                    rec = "В норме."
                    if status == "Патология":
                        if long_probs[0] > 0.6:
                            rec = "Повышен риск гипоксии в ближайшие 30 минут."
                        if long_probs[1] > 0.6:
                            rec = "Рассмотреть экстренное вмешательство."

                    result = {
                        "time": len(uc_values),  # или timestamp, если нужно
                        "short_term": {"status": status, "prob": round(s_prob, 4)},
                        "long_term": {
                            "hypoxia_30": round(float(long_probs[0]), 4),
                            "emergency_30": round(float(long_probs[1]), 4)
                        },
                        "recommendation": rec
                    }

                    await ws.send(json.dumps(result, ensure_ascii=False))
                    print("✅ Отправлено:", result)

                    last_len_uc = len(uc_values)
                    last_len_bpm = len(bpm_values)

                except Exception as e:
                    print(f"❌ Ошибка: {e}")

            await asyncio.sleep(0.1)

# --- Запуск ---
async def main():
    await asyncio.gather(listen_uc(), listen_bpm(), send_to_ai())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Остановлено")