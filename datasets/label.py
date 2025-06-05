import sys
sys.path.append("..")

import argparse
from typing import TYPE_CHECKING
from collections import Counter
import pandas as pd
from pandarallel import pandarallel
import torch
from torch.utils.data import DataLoader
import tqdm
import spacy
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DebertaV2Tokenizer
from modules.consts import PARALLELISM
from modules.utils import get_device, device_to_normalized_form, str_to_dtype
from modules.dataset import TestDataset
from modules.preprocess import preprocess


MODEL_NAME = "microsoft/deberta-v3-large"
SENTIMENT_MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"
TOKEN_LIMIT = 512 # distilbert

# 2) spaCy English 모델 로드
nlp = spacy.load("en_core_web_sm")
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME, use_fast=True)
sentiment_tokenizer = DistilBertTokenizer.from_pretrained(SENTIMENT_MODEL_NAME, use_fast=True)
sentiment_classifier = DistilBertForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)

# 기본 특성 추출: 단어 길이·어휘량·단어밀도·문장부호 등
def extract_basic_features(series: pd.Series) -> pd.Series:
    words = series['text'].split()
    series['avg_word_length'] = sum(len(word) for word in words) / len(words) if words else 0
    series['vocab_size'] = len(set(words))
    series['word_density'] = series['vocab_size'] / len(words) if words else 0
    series['lexical_diversity'] = series['word_density']
    series['question_count'] = series['text'].count(r'\?')
    series['exclamation_count'] = series['text'].count(r'!')
    series['ellipsis_count'] = series['text'].count(r'\.\.\.|…')
    return series

# 영어 특징 추출: 기능어·명사·반복·문장 길이
FUNCTION_POS = {"DET","PRON","ADP","AUX","CCONJ","SCONJ","PART"}

def extract_english_features(series: pd.Series) -> pd.Series:
    doc = nlp(series['text'])  # spaCy 언어 모델
    tokens = [tok.text for tok in doc if not tok.is_space]
    poses  = [tok.pos_  for tok in doc if not tok.is_space]

    # 기능어(function words) 개수
    func = sum(1 for p in poses if p in FUNCTION_POS)
    # 명사 개수 및 비율
    noun = sum(1 for p in poses if p in ("NOUN", "PROPN"))
    noun_ratio = noun / len(tokens) if tokens else 0
    # 토큰 중복 개수
    tok_rep = len(tokens) - len(set(tokens))
    # 바이그램 중복 개수
    bigrams = list(zip(tokens, tokens[1:]))
    bi_counts = Counter(bigrams)
    bi_rep = sum(c - 1 for c in bi_counts.values() if c > 1)
    # 문장별 단어·문자 길이 평균
    sents = list(doc.sents)
    wlens = [len([tok for tok in sent if not tok.is_space]) for sent in sents]
    avg_sw_len = sum(wlens) / len(wlens) if wlens else 0
    clens = [len(sent.text) for sent in sents]
    avg_sc_len = sum(clens) / len(clens) if clens else 0

    # 4) DataFrame에 컬럼 추가
    series['func_word_count']         = func
    series['noun_count']              = noun
    series['noun_ratio']              = noun_ratio
    series['token_repetition_count']  = tok_rep
    series['bigram_repetition_count'] = bi_rep
    series['avg_sent_len_word']       = avg_sw_len
    series['avg_sent_len_char']       = avg_sc_len

    return series

def add_sentiment(df: pd.DataFrame, dataloader: DataLoader, device: torch.device) -> pd.DataFrame:
    df['sentiment_label'] = None
    df['sentiment_score'] = None

    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(dataloader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = sentiment_classifier(input_ids=input_ids, attention_mask=attention_mask)
            outputs = torch.softmax(outputs.logits, dim=1)

            labels = torch.argmax(outputs, dim=1)
            scores = torch.max(outputs, dim=1).values
            for j in range(input_ids.size(0)):
                assert dataloader.batch_size is not None
                df.loc[i * dataloader.batch_size + j, 'sentiment_label'] = labels[j].item()
                df.loc[i * dataloader.batch_size + j, 'sentiment_score'] = scores[j].item()

    return df

# 예외처리
def remove_outliers(series: pd.Series) -> bool:
    token_count = len(tokenizer.encode(series['text'], add_special_tokens=True))
    if token_count > 768:
        return False

    if series['text'].startswith('['):
        return False

    if series['avg_word_length'] > 120:
        return False

    return True

# Usage (CPU):
# python dataset_label.py
#   --input <입력 csv 파일 경로>
#   --output <출력 경로>
#   --device cpu
#   --sentiment-batch-size 2048
# Usage (GPU):
# python label_data.py
#   --input <입력 csv 파일 경로>
#   --output <출력 경로>
#   --dtype bf16
#   --sentiment-batch-size 1024

# Example (CPU): 
# python dataset_label.py --input "./datasets/train/split/basic/train_set_human.csv" --output "./datasets/train/split/extended/train_set_human_extended.csv" --device cpu --sentiment-batch-size 2048
# python dataset_label.py --input "./datasets/train/split/basic/train_set_ai.csv" --output "./datasets/train/split/extended/train_set_ai_extended.csv" --device cpu --sentiment-batch-size 2048
# python dataset_label.py --input "./datasets/test/split/basic/test_set_human.csv" --output "./datasets/test/split/extended/test_set_human_extended.csv" --device cpu --sentiment-batch-size 2048
# python dataset_label.py --input "./datasets/test/split/basic/test_set_ai.csv" --output "./datasets/test/split/extended/test_set_ai_extended.csv" --device cpu --sentiment-batch-size 2048

# Example (GPU):
# python dataset_label.py --input "./datasets/test/split/basic/test_set_human.csv" --output "./datasets/test/split/extended/test_set_human_extended.csv" --dtype bf16 --sentiment-batch-size 1024
# python dataset_label.py --input "./datasets/test/split/basic/test_set_ai.csv" --output "./datasets/test/split/extended/test_set_ai_extended.csv" --dtype bf16 --sentiment-batch-size 1024
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--allow-outliers", action='store_true', default=False)
    parser.add_argument("--device", type=str, default=device_to_normalized_form(get_device()))
    parser.add_argument("--dtype", type=str, default="fp32")
    parser.add_argument("--sentiment-batch-size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = str_to_dtype(args.dtype)

    if device.type == "cuda":
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        torch.cuda.tunable.set_filename("tunableop.csv")

    if not TYPE_CHECKING:
        sentiment_classifier = sentiment_classifier.to(device=device, dtype=dtype)
        if device.type == "cuda":
            sentiment_classifier = torch.compile(sentiment_classifier)

    # generate _extended.csv
    pandarallel.initialize(progress_bar=True, nb_workers=PARALLELISM)
    df = pd.read_csv(args.input)
    if not TYPE_CHECKING:
        df = df.parallel_apply(extract_basic_features, axis=1)
        if not args.allow_outliers:
            df = df[df.parallel_apply(remove_outliers, axis=1)]
            df = df.reset_index(drop=True)
        df = df.parallel_apply(extract_english_features, axis=1)

    encodings = preprocess(sentiment_tokenizer, df['text'].tolist(), TOKEN_LIMIT)
    dataset = TestDataset(encodings)
    dataloader = DataLoader(dataset, batch_size=args.sentiment_batch_size, shuffle=False)
    df = add_sentiment(df, dataloader, device)

    df.to_csv(args.output, index=False)
    print(f">>> {args.output} 저장 완료")
