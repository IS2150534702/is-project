import os
from typing import Union, List, Optional
import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader
from modules.preprocess import preprocess
from modules.dataset import TestDataset
from modules.model import MultiTaskDeberta
from modules.utils import make_tokenizer


def make_dataset(raw_data: Union[os.PathLike, pd.DataFrame]) -> TestDataset:
    if isinstance(raw_data, pd.DataFrame):
        df = raw_data
    else:
        df = pd.read_csv(raw_data)

    tokenizer = make_tokenizer()
    encodings = preprocess(tokenizer, df['text'].tolist(), None)

    return TestDataset(encodings)


def evaluate(model_path: os.PathLike, raw_data: Union[os.PathLike, pd.DataFrame, TestDataset], threshold: Optional[float], batch_size: int, device: torch.device, dtype: torch.dtype):
    # 모델 불러오기
    model = MultiTaskDeberta.from_pretrained(model_path, device, dtype)
    model.eval()
    model = model.to_compiled()

    # 데이터 불러오기
    if isinstance(raw_data, TestDataset):
        dataset = raw_data
    else:
        dataset = make_dataset(raw_data)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return evaluate_impl(model, dataloader, device, threshold)


# ----------------------------------------
# 평가용 함수: 모델과 텍스트 리스트를 받아 예측값 리스트 반환
# ----------------------------------------
def evaluate_impl(model: MultiTaskDeberta, dataloader: DataLoader, device: torch.device, threshold: Optional[float]) -> List[Union[int, float]]:
    results = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)

            for j in range(len(input_ids)):
                if threshold is None:
                    result = output['main'][j].item()
                else:
                    result = 1 if output['main'][j] >= threshold else 0
                results.append(result)

    return results
