import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from modules.model import AuxiliaryDeberta
from modules.preprocess import tokenizer
from modules.utils import get_device


# ----------------------------
# 설정
# ----------------------------
MODEL_PATH = "checkpoints/model_epoch_12.pth"
DATA_PATH = "test_dataset.csv"
device = get_device()

# ----------------------------
# 모델 로딩
# ----------------------------
model = AuxiliaryDeberta()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ----------------------------
# 데이터 불러오기
# ----------------------------
df = pd.read_csv(DATA_PATH)
texts = df["text"].tolist()
labels = df["label"].tolist()

# ----------------------------
# 예측 함수
# ----------------------------
def predict_with_threshold(model, text_list, threshold):
    preds = []
    with torch.no_grad():
        for text in text_list:
            inputs = tokenizer(text, return_tensors='pt', padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(inputs["input_ids"], inputs["attention_mask"])
            prob = outputs["main"].item()
            pred = 1 if prob >= threshold else 0
            preds.append(pred)
    return preds

# ----------------------------
# Threshold 별 평가 반복
# ----------------------------
thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print("📊 Threshold별 평가 결과:\n")
for thresh in thresholds:
    preds = predict_with_threshold(model, texts, thresh)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    print(f"Threshold = {thresh:.1f} ▶ Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
