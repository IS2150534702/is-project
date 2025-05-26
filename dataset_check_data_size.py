import pandas as pd
from transformers import DebertaV2Tokenizer

MODEL_NAME = "microsoft/deberta-v3-large"
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# get csv file, 
train_basic_ai = pd.read_csv("datasets/train/split/basic/train_set_ai.csv")
train_basic_human = pd.read_csv("datasets/train/split/basic/train_set_human.csv")

print(f"train_set_ai.csv: {len(train_basic_ai)} rows")
print(f"train_set_human.csv: {len(train_basic_human)} rows")

train_extend_ai = pd.read_csv("datasets/train/split/extended/train_set_ai_extended.csv")
train_extend_human = pd.read_csv("datasets/train/split/extended/train_set_human_extended.csv")

print(f"train_set_ai_extended.csv: {len(train_extend_ai)} rows")
print(f"train_set_human_extended.csv: {len(train_extend_human)} rows")

test_set_ai = pd.read_csv("datasets/test/split/test_set_ai.csv")
test_set_human = pd.read_csv("datasets/test/split/test_set_human.csv")
print(f"test_set_ai.csv: {len(test_set_ai)} rows")
print(f"test_set_human.csv: {len(test_set_human)} rows")


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

