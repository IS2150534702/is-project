import argparse
from typing import List
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from modules.preprocess import tokenizer
from modules.model import AuxiliaryDeberta
from modules.utils import get_device


# ----------------------------------------
# 평가용 함수: 모델과 텍스트 리스트를 받아 예측값 리스트 반환
# ----------------------------------------
def predict_batch(model: AuxiliaryDeberta, texts: List[str], device: torch.device, threshold: float = 0.5):
    model.eval()
    preds = []

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, padding=True, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output = model(inputs['input_ids'], inputs['attention_mask'])
            prob = output['main'].item()
            pred = 1 if prob >= threshold else 0
            preds.append(pred)

    return preds

# ----------------------------------------
# 메인 실행
# ----------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AI Text Detection Model")
    parser.add_argument("--model", type=str, default="checkpoints/model_epoch_12.pth", help="Path to model checkpoint")
    parser.add_argument("--data", type=str, default="test_dataset.csv", help="Path to test CSV file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (default=0.5)")
    args = parser.parse_args()

    # 디바이스 설정
    device = get_device()

    # 모델 불러오기
    model = AuxiliaryDeberta()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)

    # 데이터 불러오기
    df = pd.read_csv(args.data)
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # 예측
    preds = predict_batch(model, texts, device, threshold=args.threshold)

    # 지표 계산
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    # 결과 출력
    print(f"Evaluation Results (threshold={args.threshold}):")
    print(f" - Accuracy : {acc:.4f}")
    print(f" - Precision: {prec:.4f}")
    print(f" - Recall   : {rec:.4f}")
    print(f" - F1 Score : {f1:.4f}")
