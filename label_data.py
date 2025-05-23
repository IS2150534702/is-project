import argparse
from typing import TYPE_CHECKING
from collections import Counter
import pandas as pd
from pandarallel import pandarallel
import torch
import tqdm
import spacy
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from modules.consts import PARALLELISM
from modules.utils import get_device, round_up


MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"

device = get_device()

# 2) spaCy English 모델 로드
nlp = spacy.load("en_core_web_sm")
sentiment_tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
sentiment_classifier = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)

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
    """
    df: 원본 DataFrame. 반드시 'text' 컬럼을 포함해야 합니다.
    row_limit: 처리할 최대 행 개수. None인 경우 전체를 처리합니다.
    반환: feature 컬럼이 추가된 DataFrame (최대 row_limit 행)
    """

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

def add_sentiment(df: pd.DataFrame, batch_size = 512) -> pd.DataFrame:
    df['sentiment_label'] = None
    df['sentiment_score'] = None

    for i in tqdm.trange(round_up(len(df), batch_size)):
        batched_df = df.iloc[i * batch_size:(i + 1) * batch_size]
        batched_tokens = sentiment_tokenizer(batched_df['text'].tolist(), truncation=True, padding=True, return_tensors="pt")
        input_ids = batched_tokens['input_ids'].to(device)
        attention_mask = batched_tokens['attention_mask'].to(device)
        with torch.no_grad():
            outputs = sentiment_classifier(input_ids=input_ids, attention_mask=attention_mask)
            outputs = torch.softmax(outputs.logits, dim=1)

        labels = torch.argmax(outputs, dim=1)
        scores = torch.max(outputs, dim=1).values
        for j in range(len(batched_df)):
            df.loc[i * batch_size + j, 'sentiment_label'] = labels[j].item()
            df.loc[i * batch_size + j, 'sentiment_score'] = scores[j].item()

    return df

def process_file(input_csv: str, output_csv: str):
    pandarallel.initialize(progress_bar=True, nb_workers=PARALLELISM)
    df = pd.read_csv(input_csv)
    if not TYPE_CHECKING:
        df = df.parallel_apply(extract_basic_features, axis=1)
        df = df.parallel_apply(extract_english_features, axis=1)
        df = add_sentiment(df)
    df.to_csv(output_csv, index=False)
    print(f">>> {output_csv} 저장 완료")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--destination", type=str, required=True)
    args = parser.parse_args()

    sentiment_classifier = sentiment_classifier.to(device)
    if device.type == "cuda" and not TYPE_CHECKING:
        sentiment_classifier = torch.compile(sentiment_classifier)

    process_file(args.source, args.destination)
