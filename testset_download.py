import pandas as pd
from datasets import load_dataset


# 1) 데이터셋 로드
ds = load_dataset("artnitolog/llm-generated-texts")

# 2) Pandas DataFrame 변환 및 'essay' 필터링
df = pd.DataFrame(ds['train'])
df_essay = df[df['dataset_name'] == 'essay']

# 3) Human 데이터셋 생성 (label=0)
human_df = df_essay[['human']].copy()                   
human_df = human_df.rename(columns={'human': 'text'})
human_df['label'] = 0

# 4) AI 데이터셋 생성 (label=1)
ai_fields = [
    'GPT4 Turbo 2024-04-09', 'GPT4 Omni', 'Claude 3 Opus',
    'YandexGPT 3 Pro', 'GigaChat Pro', 'Llama3 70B', 'Command R+'
]
# melt를 써서 각 모델별 컬럼을 개별 행으로 펼치기
ai_df = (
    df_essay[ai_fields]
    .melt(value_vars=ai_fields, var_name="model", value_name="text")
    .drop(columns=["model"])
    .dropna(subset=["text"]) # NaN 제거
    .reset_index(drop=True)
    .assign(label=1)
)
ai_df['label'] = 1

print(f"Human 데이터셋 길이: {len(human_df)}")
print(f"AI 데이터셋 길이: {len(ai_df)}")
# 5) CSV로 저장
human_df.to_csv('./datasets/test/split/test_set_human.csv', index=False)  # human CSV 생성[5]
ai_df.to_csv('./datasets/test/split/test_set_ai.csv', index=False)        # AI CSV 생성[5]

print(">> success")
