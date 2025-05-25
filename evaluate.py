import argparse
from typing import List, TYPE_CHECKING
import pandas as pd
import torch
from torch.utils.data import DataLoader
import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from modules.preprocess import preprocess
from modules.model import AuxiliaryDeberta
from modules.dataset import TestDataset
from modules.utils import get_device, str_to_dtype


# ----------------------------------------
# 평가용 함수: 모델과 텍스트 리스트를 받아 예측값 리스트 반환
# ----------------------------------------
def predict_batch(model: AuxiliaryDeberta, dataloader: DataLoader, device: torch.device, threshold: float = 0.5) -> List[int]:
    preds = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)

            for j in range(len(input_ids)):
                pred = 1 if output['main'][j] >= threshold else 0
                preds.append(pred)

    return preds

# ----------------------------------------
# 메인 실행
# ----------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AI Text Detection Model")
    parser.add_argument("--model", type=str, default="checkpoints/model_epoch_12.pth", help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="test_dataset.csv", help="Path to test CSV file")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="fp32")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (default=0.5)")
    args = parser.parse_args()

    # 디바이스 설정
    device = get_device()
    dtype = str_to_dtype(args.dtype)

    if device.type == "cuda":
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        torch.cuda.tunable.set_filename("tunableop.csv")

    # 모델 불러오기
    model = AuxiliaryDeberta()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device=device, dtype=dtype)
    if device.type == "cuda" and not TYPE_CHECKING:
        model = torch.compile(model)
    model.eval()

    # 데이터 불러오기
    df = pd.read_csv(args.dataset)

    labels = df['label'].tolist()
    encodings = preprocess(df['text'].tolist(), None)
    dataset = TestDataset(encodings)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 예측
    preds = predict_batch(model, dataloader, device, args.threshold)

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
