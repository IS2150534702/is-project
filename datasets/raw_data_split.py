import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from transformers import DebertaV2Tokenizer

MODEL_NAME = "microsoft/deberta-v3-large"
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))

def save_data(df: pd.DataFrame, filename: str) -> None:
    # df = df.apply(remove_outliers, axis=1)
    df.to_csv(filename, index=False)
    print(f"{filename} 저장 완료: {len(df)} rows")

def preprocess_token_count(df: pd.DataFrame) -> pd.DataFrame:
    df["token_count"] = df["text"].apply(count_tokens)
    df = df[df["token_count"] <= 768]
    df = df.drop(columns=["token_count"])
    return df[['text', 'label']].copy()

def split_raw_by_label(input: str, same_ratio: bool = False) -> None:
    # 2) raw_data.csv 읽어서 label별로 분리 후 train/test 0.8/0.2 분할
    raw = pd.read_csv(input)

    if same_ratio:
        human = raw[raw['label'] == 0]
        ai    = raw[raw['label'] == 1]

        h_train, h_test = train_test_split(
            human, test_size=0.2, random_state=42, shuffle=True)
        a_train, a_test = train_test_split(
            ai,    test_size=0.2, random_state=42, shuffle=True)

        save_data(h_train, './train/split/basic/train_set_human.csv')
        save_data(a_train, './train/split/basic/train_set_ai.csv')
        save_data(h_test,  './test/split/test_set_human.csv')
        save_data(a_test,  './test/split/test_set_ai.csv')
    else:
        # 2) 전체 데이터를 80% train / 20% test로 분할
        train_df, test_df = train_test_split(
            raw, test_size=0.2, random_state=42, shuffle=True
        )
        # 3) train_set 저장: label=0(human), label=1(ai)
        save_data(train_df[train_df['label'] == 0], './train/split/basic/train_set_human.csv')
        save_data(train_df[train_df['label'] == 1], './train/split/basic/train_set_ai.csv')
        save_data(test_df[test_df['label'] == 0], './test/split/test_set_human.csv')
        save_data(test_df[test_df['label'] == 1], './test/split/test_set_ai.csv')

    human_test = pd.read_csv('./test/split/test_set_human.csv')
    ai_test  = pd.read_csv('./test/split/test_set_ai.csv')

    human_test = preprocess_token_count(human_test)
    ai_test = preprocess_token_count(ai_test)

    combined = pd.concat([ai_test, human_test], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    combined.to_csv('./test/test_set_final.csv', index=False)
    print(f"./test/test_set_final.csv 저장 완료: {len(combined)} rows")

# Usage:
# python raw_data_split.py  --input raw_data.csv --same_ratio=False
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--same_ratio", type=bool, default=False)

    args = parser.parse_args()
    if args.same_ratio:
        split_raw_by_label(args.input)
    else:
        split_raw_by_label(args.input)
