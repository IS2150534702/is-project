import pandas as pd

# 데이터 불러오기
human_df = pd.read_csv("./train/split/extended/train_set_human_extended.csv")
ai_df = pd.read_csv("./train/split/extended/train_set_ai_extended.csv")

print(f"human_df 길이: {len(human_df)}")
print(f"ai_df 길이: {len(ai_df)}")

# 데이터 합치기
df = pd.concat([human_df, ai_df], ignore_index=True)
# df = ai_df

# 전체 데이터셋에서의 상관계수 (참고용)
correlation_matrix_full = df[['avg_word_length', 'func_word_count']].corr()
correlation_value_full = correlation_matrix_full.loc['avg_word_length', 'func_word_count']
print(f"\n전체 데이터셋에서의 상관계수: {correlation_value_full}")

# 상관관계 계산 방법 설명 (피어슨 상관계수)
# print("\n사용한 상관관계 분석 방법 (샘플 데이터):")
# print("위 코드에서는 pandas 라이브러리의 '.corr()' 메소드를 사용하여 샘플 데이터의 상관계수를 계산했습니다.")
# print("이 메소드는 기본적으로 피어슨 상관계수(Pearson correlation coefficient)를 계산합니다.")

# 나머지 코드 진행 (원래 요청대로 aux1, aux2로 이름 변경 및 최종 저장)
df_final = df[['text', 'label', 'avg_word_length', 'func_word_count']].copy()
df_final.rename(columns={
    'avg_word_length': 'aux1',
    'func_word_count': 'aux2'
}, inplace=True)
# 최종 데이터셋은 전체 데이터를 사용하여 섞습니다.
df_final = df_final.sample(frac=1, random_state=128).reset_index(drop=True)

# print(f"\n합쳐진 데이터프레임 길이 (최종 저장용): {len(df_final)}")
# df_final.to_csv("./datasets/train/train_set_final.csv", index=False)  # 결과 저장
# print("\n최종 데이터셋이 './datasets/train/train_set_final.csv'에 저장되었습니다.")
