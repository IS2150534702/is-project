import os
import argparse
from typing import Optional, Dict, Tuple, List, Any, TYPE_CHECKING
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DebertaV2Tokenizer
import pandas as pd
from pandarallel import pandarallel
from sklearn.preprocessing import MinMaxScaler
from modules import consts
from modules.model import AuxiliaryDeberta
from modules.loss import compute_multitask_loss
from modules.dataset import TrainDataset
from modules.preprocess import preprocess_for_train
from modules.utils import get_device, str_to_dtype


def preprocess_data(x, features_scaled: np.ndarray) -> Dict[str, Any]:
    return {
        'main': float(x['label']),
        'aux': [float(feature) for feature in features_scaled[x.name]],
    }

# CSV 파일을 읽고, 정규화까지 처리하는 함수
def load_and_preprocess_data(path: os.PathLike, scaler: MinMaxScaler, fit_scaler=False) -> Tuple[List[str], List[Dict[str, Any]]]:
    df = pd.read_csv(path)
    texts = df['text'].tolist()
    features = df[['aux1', 'aux2']].copy()

    if fit_scaler:
        scaler.fit(features)
    features_scaled = scaler.transform(features)

    labels = []
    if not TYPE_CHECKING:
        labels = df.parallel_apply(lambda x: preprocess_data(x, features_scaled), axis=1)

    return texts, labels

# 모델이 한 번 훈련 혹은 검증을 수행하는 함수
def run_epoch(model: AuxiliaryDeberta, dataloader: DataLoader, optimizer: Optional[torch.optim.Optimizer] = None, weights: List[float] = [1.0, *([0.0] * AuxiliaryDeberta.NUM_AUXILIARY_TASKS)]) -> float:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    for batch in tqdm.tqdm(dataloader, leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = {k: v.to(device) for k, v in batch['labels'].items()}

        with torch.set_grad_enabled(is_train):
            if is_train:
                optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss, _ = compute_multitask_loss(outputs, labels, weights)

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AI Text Detection Model")
    parser.add_argument("--path", type=str, help="Path to checkpoint file", required=False)
    parser.add_argument("--train", type=str, default="train_dataset.csv", help="Path to train CSV file")
    parser.add_argument("--val", type=str, help="Path to validate CSV file", required=False)
    parser.add_argument("--weights", type=str, required=False, default="1.0")
    parser.add_argument("--epoch", type=int, help="Epoch", required=True)
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--dtype", type=str, default="fp32", help="DataType")
    args = parser.parse_args()

    pandarallel.initialize(progress_bar=True, nb_workers=consts.PARALLELISM)

    # 디바이스 설정
    device = get_device()
    dtype = str_to_dtype(args.dtype)

    if device.type == "cuda":
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        torch.cuda.tunable.set_filename("tunableop.csv")

    # 정규화 도구 생성
    scaler = MinMaxScaler()

    tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-large', use_fast=True)

    # 훈련 데이터 전처리
    train_texts, train_labels = load_and_preprocess_data(args.train, scaler, fit_scaler=True)
    # 텍스트 전처리
    train_encodings = preprocess_for_train(tokenizer, train_texts, train_labels, None, dtype)
    # Dataset 구성
    train_dataset = TrainDataset(train_encodings)
    # DataLoader 설정
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    weights = [None if len(weight) == 0 else float(weight) for weight in args.weights.split(",")]
    for i in range(len(weights), AuxiliaryDeberta.NUM_AUXILIARY_TASKS + 1):
        weights.append(None)
    # 모델 및 옵티마이저 초기화
    if args.path is None:
        model = AuxiliaryDeberta(auxiliary_tasks=[weight is not None for weight in weights[1:]])
    else:
        model = AuxiliaryDeberta.from_state_dict(torch.load(args.path, map_location=device))
        for i, weight in enumerate(weights):
            if weight is not None and not model.has_auxiliary_task(i):
                raise ValueError(f"Checkpoint does not contain auxiliary task #{i}.")
    model = model.to(device=device, dtype=dtype)
    model_runtime = model
    if device.type == "cuda" and not TYPE_CHECKING:
        model_runtime = torch.compile(model)
    optimizer = torch.optim.AdamW(model_runtime.parameters(), lr=2e-5)

    # 체크포인트 디렉토리 생성
    os.makedirs("checkpoints", exist_ok=True)

    print(f"PyTorch: device={device}, dtype={dtype}")
    print(f"Auxiliary tasks enabled: {', '.join([f'aux{i}={weight}' for i, weight in enumerate(weights) if i != 0])}")

    weights = [0.0 if weight is None else weight for weight in weights]

    progress = tqdm.trange(args.epoch, desc=f"Epoch 0/{args.epoch}")
    if args.val is None:
        # 학습 루프
        progress.set_postfix(loss='?')
        for epoch in progress:
            loss = run_epoch(model_runtime, train_loader, optimizer, weights)
            progress.set_description(f"Epoch {epoch + 1}/{args.epoch}")
            progress.set_postfix(loss=loss)
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch}.pth")
    else:
        val_texts, val_labels = load_and_preprocess_data(args.val, scaler)
        val_encodings = preprocess_for_train(tokenizer, val_texts, val_labels, None, dtype)
        val_dataset = TrainDataset(val_encodings)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        # 학습 루프
        progress.set_postfix(train_loss='?', val_loss='?')
        for epoch in progress:
            train_loss = run_epoch(model_runtime, train_loader, optimizer)
            val_loss = run_epoch(model_runtime, val_loader)
            progress.set_description(f"Epoch {epoch + 1}/{args.epoch}")
            progress.set_postfix(train_loss=train_loss, val_loss=val_loss)
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch}.pth")
