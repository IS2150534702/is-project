import os
from typing import Union, List
import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import DebertaV2Tokenizer
from modules.preprocess import preprocess
from modules.dataset import TestDataset
from modules.model import AuxiliaryDeberta


def evaluate(model_path: os.PathLike, raw_data: Union[pd.DataFrame, os.PathLike], threshold: float, batch_size: int, device: torch.device, dtype: torch.dtype):
    # 모델 불러오기
    model = AuxiliaryDeberta.from_pretrained(model_path, device, dtype, device.type == "cuda")
    model.eval()

    # 데이터 불러오기
    if isinstance(raw_data, pd.DataFrame):
        df = raw_data
    else:
        df = pd.read_csv(raw_data)

    tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-large', use_fast=True)
    encodings = preprocess(tokenizer, df['text'].tolist(), None)
    dataset = TestDataset(encodings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 예측
    return evaluate_impl(model, dataloader, device, threshold)


# ----------------------------------------
# 평가용 함수: 모델과 텍스트 리스트를 받아 예측값 리스트 반환
# ----------------------------------------
def evaluate_impl(model: AuxiliaryDeberta, dataloader: DataLoader, device: torch.device, threshold: float) -> List[int]:
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
