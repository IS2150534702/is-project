import pandas as pd
from transformers import DebertaV2Tokenizer

MODEL_NAME = "microsoft/deberta-v3-large"
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))

# 1) 데이터셋 로드
human_df = pd.read_csv("./datasets/test/split/test_set_human.csv")
ai_df = pd.read_csv("./datasets/test/split/test_set_ai.csv")

print(f"human_df 길이: {len(human_df)}")
print(f"ai_df 길이: {len(ai_df)}")

df = pd.concat([human_df, ai_df], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)
print(f"합쳐진 데이터프레임 길이: {len(df)}")

# df["token_count"] = df["text"].apply(count_tokens)
# # 2) 토큰 수가 768 이하인 데이터만 필터링
# df = df[df["token_count"] <= 768]
# df = df.drop(columns=["token_count"])

df = df[['text', 'label']].copy()
print(f"선택된 데이터프레임 길이: {len(df)}")

df.to_csv("./datasets/test/test_set_final.csv", index=False)  # 결과 저장