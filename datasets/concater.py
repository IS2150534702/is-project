import pandas as pd

human_df = pd.read_csv("./train/split/extended/train_set_human_extended.csv")
ai_df = pd.read_csv("./train/split/extended/train_set_ai_extended.csv")

# human_df = human_df[['text', 'label', 'func_word_count', 'token_repetition_count']].copy()
# ai_df = ai_df[['text', 'label', 'func_word_count', 'token_repetition_count']].copy()

print(f"human_df 길이: {len(human_df)}")
print(f"ai_df 길이: {len(ai_df)}")

df = pd.concat([human_df, ai_df], ignore_index=True)
df = df[['text', 'label', 'avg_word_length', 'func_word_count']].copy()
df.rename(columns={
    'avg_word_length': 'aux1',
    'func_word_count': 'aux2'
}, inplace=True)
df = df.sample(frac=1).reset_index(drop=True)

print(f"합쳐진 데이터프레임 길이: {len(df)}")
df.to_csv("./train/train_set_final.csv", index=False)  # 결과 저장