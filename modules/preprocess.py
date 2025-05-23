from typing import List, Dict
from transformers import DebertaV2Tokenizer
import torch
from pqdm.threads import pqdm
from modules import consts


tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-large', use_fast=True)
def preprocess(texts: List[str], labels: List[Dict[str, float]], dtype=torch.float32) -> Dict[str, torch.Tensor]:
    encodings = pqdm(texts, lambda x: tokenizer(x), n_jobs=consts.PARALLELISM, max_workers=consts.PARALLELISM)
    encodings = tokenizer.pad(encodings, padding=True, return_tensors="pt")
    encodings['main'] = torch.tensor([label['main'] for label in labels], dtype=dtype)
    encodings['aux1'] = torch.tensor([label['aux1'] for label in labels], dtype=dtype)
    encodings['aux2'] = torch.tensor([label['aux2'] for label in labels], dtype=dtype)
    encodings['aux3'] = torch.tensor([label['aux3'] for label in labels], dtype=dtype)
    return encodings
