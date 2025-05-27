from typing import List, Dict, Any, Optional
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils import PreTrainedTokenizer
import torch
from pqdm.threads import pqdm
from modules import consts


def preprocess_for_train(tokenizer: PreTrainedTokenizer, texts: List[str], labels: List[Dict[str, Any]], truncation: Optional[int] = None, dtype = torch.float32) -> BatchEncoding:
    encodings = pqdm(texts, lambda x: tokenizer(x) if truncation is None else tokenizer(x, truncation=True, max_length=truncation), n_jobs=consts.PARALLELISM, max_workers=consts.PARALLELISM)
    encodings = tokenizer.pad(encodings, padding=True, return_tensors="pt")
    encodings['main'] = torch.tensor([label['main'] for label in labels], dtype=dtype)
    encodings['aux'] = [torch.tensor([label['aux'][i] for label in labels], dtype=dtype) for i in range(len(labels[0]['aux']))]
    return encodings

def preprocess(tokenizer: PreTrainedTokenizer, texts: List[str], truncation: Optional[int] = None) -> BatchEncoding:
    encodings = pqdm(texts, lambda x: tokenizer(x) if truncation is None else tokenizer(x, truncation=True, max_length=truncation), n_jobs=consts.PARALLELISM, max_workers=consts.PARALLELISM)
    encodings = tokenizer.pad(encodings, padding=True, return_tensors="pt")
    return encodings
