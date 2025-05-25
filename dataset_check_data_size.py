import pandas as pd
from transformers import DebertaV2Tokenizer

MODEL_NAME = "microsoft/deberta-v3-large"
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# get csv file, 
df = pd.read_csv("datasets/train/split/extended/train_set_ai_extended.csv")
df2 = pd.read_csv("datasets/train/split/extended/train_set_human_extended.csv")

# def count_tokens(text: str) -> int:
#     return len(tokenizer.encode(text, add_special_tokens=True))

# ai_extended = df.copy()
# human_extended = df2.copy()

# ai_counts = ai_extended['text'].apply(count_tokens)
# human_counts  = human_extended['text'].apply(count_tokens)

# ai_counts = ai_counts.tolist()
# human_list = human_counts.tolist()
# for i in range(len(ai_counts)):
#     if ai_counts[i] > 768:
#         print(f"Index {i} has token count {ai_counts[i]} which is greater than 768.")

df3 = pd.read_csv("datasets/train/train_set_final.csv")
df4 = pd.read_csv("datasets/test/test_set_final.csv")
df5 = pd.read_csv("datasets/test/test_set_final_under_768.csv")

ai_cnt = df5['label'] == 1
human_cnt = df5['label'] == 0
print(f"AI count: {ai_cnt.sum()}")
print(f"Human count: {human_cnt.sum()}")

print(len(df))
print(len(df2))
print(len(df3))
print(len(df4))
print(len(df5))

