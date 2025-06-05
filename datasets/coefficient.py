import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 데이터 불러오기 (이전 코드와 동일)
human_df = pd.read_csv("./train/split/extended/train_set_human_extended.csv")
ai_df = pd.read_csv("./train/split/extended/train_set_ai_extended.csv")
df = pd.concat([human_df, ai_df], ignore_index=True)

# 분석을 위해 결측값이나 부적절한 값 제거 (예시)
df_cleaned = df[['avg_word_length', 'func_word_count']].copy()
df_cleaned.dropna(inplace=True)

print("--- 1. 로그-로그 회귀 분석 ---")
# 로그-로그 회귀를 위해서는 변수들이 양수여야 합니다.
df_log_log = df_cleaned[(df_cleaned['avg_word_length'] > 0) & (df_cleaned['func_word_count'] > 0)].copy()

if not df_log_log.empty:
    df_log_log['log_avg_word_length'] = np.log(df_log_log['avg_word_length'])
    df_log_log['log_func_word_count'] = np.log(df_log_log['func_word_count'])

    # 독립 변수(X)와 종속 변수(Y) 설정
    # func_word_count를 종속 변수로 가정합니다.
    Y_log = df_log_log['log_func_word_count']
    X_log = df_log_log['log_avg_word_length']
    X_log = sm.add_constant(X_log) # 절편(상수항) 추가

    model_log_log = sm.OLS(Y_log, X_log)
    results_log_log = model_log_log.fit()

    print("\n로그-로그 회귀 모델 요약:")
    print(results_log_log.summary())
    print("\n설명:")
    print("로그-로그 모델은 독립 변수와 종속 변수 모두에 로그 변환을 적용합니다[1, 5].")
    print("이 모델에서 독립 변수의 계수(coef)는 탄력성(elasticity)으로 해석될 수 있습니다.")
    print("예를 들어, log_avg_word_length의 계수가 -0.5라면, avg_word_length가 1% 증가할 때 func_word_count가 약 0.5% 감소하는 경향이 있음을 의미합니다.")
    print("R-squared 값은 모델이 종속 변수의 변동성을 얼마나 설명하는지를 나타냅니다.")
else:
    print("\n로그-로그 회귀 분석을 위한 충분한 양의 데이터가 없습니다 (모든 값이 0 이하일 수 있음).")


print("\n\n--- 2. 스피어만 상관계수 ---")
if not df_cleaned.empty:
    spearman_corr, p_value_spearman = spearmanr(df_cleaned['avg_word_length'], df_cleaned['func_word_count'])
    # Pandas DataFrame의 .corr(method='spearman')도 사용 가능합니다:
    # spearman_corr_pandas = df_cleaned[['avg_word_length', 'func_word_count']].corr(method='spearman').iloc[0,1]

    print(f"\n스피어만 순위 상관계수 (avg_word_length vs func_word_count): {spearman_corr:.4f}")
    print(f"P-value: {p_value_spearman:.4f}")

    print("\n설명:")
    print("스피어만 상관계수(Spearman's rank correlation coefficient)는 두 변수 간의 단조 관계(monotonic relationship)를 측정하는 비모수적 방법입니다[2].")
    print("값이 -1에서 1 사이이며, 1에 가까울수록 강한 양의 단조 관계, -1에 가까울수록 강한 음의 단조 관계를 나타냅니다.")
    print("0에 가까우면 단조 관계가 거의 없음을 의미합니다[6].")
    print("이 계수는 변수들의 실제 값이 아닌 순위(rank)를 사용하여 계산하므로, 피어슨 상관계수와 달리 선형 관계를 가정하지 않습니다[2].")
    print("P-value는 계산된 상관계수가 통계적으로 유의미한지를 판단하는 데 사용됩니다 (일반적으로 0.05보다 작으면 유의미하다고 봅니다).")
else:
    print("\n스피어만 상관계수 분석을 위한 데이터가 없습니다.")


print("\n\n--- 3. 비선형 모델 (예: 다항 회귀) ---")
if not df_cleaned.empty and len(df_cleaned) > 5: # 충분한 데이터가 있는지 확인
    X_poly_data = df_cleaned[['avg_word_length']]
    y_poly_data = df_cleaned['func_word_count']

    # 데이터를 훈련 세트와 테스트 세트로 분리 (모델 평가를 위함)
    X_train, X_test, y_train, y_test = train_test_split(X_poly_data, y_poly_data, test_size=0.2, random_state=42)

    # 2차 다항 특성 생성
    degree = 2
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    # 다항 회귀 모델 학습
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)

    # 테스트 데이터로 예측 및 평가
    y_pred_poly = poly_model.predict(X_test_poly)
    r2_poly = r2_score(y_test, y_pred_poly)

    print(f"\n다항 회귀 (차수={degree}) 모델 R-squared (테스트 데이터): {r2_poly:.4f}")
    print(f"회귀 계수: {poly_model.coef_}")
    print(f"절편: {poly_model.intercept_}")

    print("\n설명:")
    print("다항 회귀는 독립 변수의 다항식(예: x, x^2, x^3 등)을 사용하여 종속 변수와의 비선형 관계를 모델링합니다[3, 4].")
    print(f"여기서는 {degree}차 다항식을 사용했습니다. 모델은 Y = b0 + b1*X + b2*X^2 형태를 가정합니다.")
    print("R-squared 값은 이 다항 모델이 테스트 데이터에서 종속 변수의 변동성을 얼마나 잘 설명하는지 나타냅니다.")
    print("다항 회귀는 변수 변환을 통해 선형 회귀의 틀 내에서 비선형 관계를 유연하게 모델링할 수 있게 합니다[4].")
    print("주의: 다항식의 차수가 너무 높아지면 과적합(overfitting)의 위험이 있습니다.")
else:
    print("\n다항 회귀 분석을 위한 충분한 데이터가 없습니다.")


# 나머지 코드 (최종 데이터셋 저장 등)는 필요에 따라 여기에 추가할 수 있습니다.
# df_final = df[['text', 'label', 'avg_word_length', 'func_word_count']].copy()
# df_final.rename(columns={
# 'avg_word_length': 'aux1',
# 'func_word_count': 'aux2'
# }, inplace=True)
# df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
# df_final.to_csv("./datasets/train/train_set_final.csv", index=False)
# print("\n최종 데이터셋 저장 완료 (필요시 주석 해제).")

