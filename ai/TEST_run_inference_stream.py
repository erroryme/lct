import os, torch, joblib, json
import numpy as np
import pandas as pd
from TEST_train_resnet_multihead import MultiHeadMLP, device

BASE = os.path.dirname(__file__)

# === Load ===
df = pd.read_csv(os.path.join(BASE,"features_unified.csv"))
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

# Берём только числовые признаки
feat_cols = [c for c in df.columns if c not in ["label","hypoxia_30","emergency_30"]
             and pd.api.types.is_numeric_dtype(df[c])]

X = df[feat_cols].values.astype(np.float32)
y = df["label"].values.astype(np.int64) if "label" in df.columns else None


scaler = joblib.load(os.path.join(BASE,"scaler_unified.pkl"))
X = scaler.transform(X)

model = MultiHeadMLP(in_dim=X.shape[1]).to(device)
model.load_state_dict(torch.load(os.path.join(BASE,"unified_mlp.pt"), map_location=device))
model.eval()

with open(os.path.join(BASE,"thresholds.json")) as f:
    thresholds=json.load(f)
print(f"Используем thresholds: {thresholds}")

# === Inference Stream (увеличили n_samples до 200) ===
n_samples=200
step=180_000
statuses=[]; hypox=[]; emerg=[]

for i in range(n_samples):
    idx=i%len(X)
    xb=torch.tensor(X[idx:idx+1]).to(device)
    with torch.no_grad():
        s_logit,long_out=model(xb)
        s_prob=torch.sigmoid(s_logit).cpu().item()
        long_probs=torch.sigmoid(long_out).cpu().numpy().ravel()

    status="Норма"
    if s_prob>thresholds["t_high"]: status="Патология"
    elif s_prob>thresholds["t_mid"]: status="Подозрение"

    rec="В норме."
    if status=="Патология":
        if long_probs[0]>0.6: rec="Повышен риск гипоксии в ближайшие 30 минут."
        if long_probs[1]>0.6: rec="Рассмотреть экстренное вмешательство."

    res={
        "time":i*step,
        "short_term":{"status":status,"prob":round(s_prob,4)},
        "events":[], # тут можно дополнять эвенты
        "long_term":{"hypoxia_30":round(float(long_probs[0]),4),"emergency_30":round(float(long_probs[1]),4)},
        "recommendation":rec
    }
    statuses.append(status); hypox.append(long_probs[0]); emerg.append(long_probs[1])
    print(res)

print("\n📊 Summary:")
from collections import Counter
print("Статусы:",Counter(statuses))
print(f"Средний hypoxia_30={np.mean(hypox):.3f}, emergency_30={np.mean(emerg):.3f}")
