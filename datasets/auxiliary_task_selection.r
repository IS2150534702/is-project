library(readr)


# 전처리 작업 및 라벨링을 마친 데이터셋 불러오기
# show_col_types = FALSE로 지정함으로써 각 열의 자료형은 표시 X
human_train <- read_csv("./train/split/extended/train_set_human_extended.csv", show_col_types = FALSE)
ai_train <- read_csv("./train/split/extended/train_set_ai_extended.csv", show_col_types = FALSE)
View(human_train)
View(ai_train)

# human에서 text를 제외한 열 선택
human_subset <- data.frame(human_train[, 2:ncol(human_train)])
head(human_subset)

# ai에서 text를 제외한 열 선택
ai_subset <- data.frame(ai_train[, 2:ncol(ai_train)])
head(ai_subset)

# 행 기준으로 합치기 (위아래로 붙임)
combined_subset <- rbind(human_subset, ai_subset)

# 결과 확인
View(combined_subset)








##########    요약 통계 비교
library(dplyr)

# 1. 비교할 열 추출 (label 열인 1열을 제외한 2열부터)
feature_names <- colnames(combined_subset)[2:ncol(combined_subset)]

# 2) 각 feature에 대해 기본적인 방식으로 summary 계산
for (feat in feature_names) {
  cat("=====================================\n")
  cat(paste0("Feature: ", feat, "\n\n"))
  
  # Human & AI 그룹으로 추출 (drop=TRUE : 열 하나를 선택하되 데이터 프레임 유지)
  h <- combined_subset[combined_subset$label == 0, feat, drop = TRUE]
  a <- combined_subset[combined_subset$label == 1, feat, drop = TRUE]
  
  # 두 그룹을 한 data.frame으로 정리
  df <- data.frame(
    label  = c(0, 1),
    Min     = c(min(h,   na.rm = TRUE), min(a,   na.rm = TRUE)),
    Q1      = c(quantile(h, 0.25, na.rm = TRUE), quantile(a, 0.25, na.rm = TRUE)),
    Median  = c(median(h, na.rm = TRUE),       median(a, na.rm = TRUE)),
    Mean    = c(mean(h,   na.rm = TRUE),        mean(a,   na.rm = TRUE)),
    Q3      = c(quantile(h, 0.75, na.rm = TRUE), quantile(a, 0.75, na.rm = TRUE)),
    Max     = c(max(h,   na.rm = TRUE),         max(a,   na.rm = TRUE))
  )
  
  print(df)
  cat("\n\n")
}



# t-test와 F-test는 정규분포를 가정하는 검정이지만,
# 본 프로젝트에서 train_dataset으로는 각 그룹의 데이터 수(train_Ai : 11284, train_Human : 21382)가 수만 건 이상으로 매우 많기 때문에,
# 중심극한정리에 따라 정규성 가정에 크게 의존하지 않아도 됨.
# 두 집단의 분산이 같은지 여부에 따라 t-test의 버전(Student vs. Welch)을 정확하게 선택하기 위해 var.test(분산성을 확인)를 실시
# sentiment_label은 범주형 변수이고, question_count와 ellipsis_count는 두 집단에서 모두 값이 0이므로 비교가 불가능함.
# 따라서, 모든 feature에 대해서 등분산성 및 t-검정 코드를 작성했으나, 9개의 feature의 결과에만 집중해야 한다.
# 9개의 feature에 대해 var.test()시 p-value가 0.05보다 작으므로 두 집단의 분산이 다르기에 welch's t-test를 적용 (등분산의 경우 Student's t-test를 적용해야 함)

# welch's t-test를 통해 출력한 결과에서 평균차이가 큰지, p-value를 통해 통계적으로 유의한지를 따져보아 다음과 같은 결론을 도출
# 1. avg_word_length
# 2. func_word_count


########## 등분산성 검정 및 t-검정
features <- feature_names

for (feat in features) {
  cat("=====", feat, "=====\n")
  
  # 두 그룹 벡터 추출
  x <- combined_subset[combined_subset$label == 0, feat, drop = TRUE]
  y <- combined_subset[combined_subset$label == 1, feat, drop = TRUE]
  
  # 등분산성 검정 (F-test)
  ftest <- tryCatch(var.test(x, y), error = function(e) NA)
  if (is.list(ftest)) {
    print(ftest)
  } else {
    cat("⚠️ F-test 계산 불가 (분산 없음 등)\n")
  }
  
  # t-test: F-test 결과에 따라 선택
  if (is.list(ftest) && !is.na(ftest$p.value)) {
    if (ftest$p.value < 0.05) {
      cat("→ 분산 다름: Welch's t-test 적용\n")
      ttest <- t.test(x, y, var.equal = FALSE)
    } else {
      cat("→ 분산 같음: Student's t-test 적용\n")
      ttest <- t.test(x, y, var.equal = TRUE)
    }
  } else {
    cat("→ F-test 실패: Welch's t-test 강제 적용\n")
    ttest <- t.test(x, y, var.equal = FALSE)
  }
  
  # t-test 출력
  if (is.finite(ttest$statistic)) {
    print(ttest)
  } else {
    cat("⚠️ 평균 비교 불가: 데이터 값이 동일하거나 표준편차 없음\n")
  }
  
  cat("\n")
}

########## 결론 : 
# 시각화 자료에서도 그렇고 t-test 결과 두 집단 (ai와 human)의 평균에 대해 유의미한 차이가 있음이 검증된 feature로는 avg_word_length, func_word_count가 후보로 여겨짐
# 따라서, 많은 데이터 개수에 의해 중심극한정리에 따라 정규성을 만족한다고 할 수 있으며, 등분산성을 평가에 의해 두 집단간 분산성이 존재하는 feature에 대해
# Welch's t-test까지도 통과했기에 avg_word_length, func_word_count는 보조 태스크로 적합하다고 판단.











########### 모든 column에 대해 시각화 :
features <- c("avg_word_length","vocab_size","word_density","lexical_diversity","exclamation_count","func_word_count","noun_count","noun_ratio","token_repetition_count","bigram_repetition_count","avg_sent_len_word","avg_sent_len_char","sentiment_label","sentiment_score")    # t-"func_word_count","token_repetition_count","avg_word_length" / auc-"avg_word_length", "sentiment_score", "func_word_count"

# 1) Boxplot 출력
par(mfrow = c(3, 5), mar = c(4, 4, 2, 1))  # 3행 5열 (15칸 중 1개는 비어 있음)
for (feat in features) {
  h <- combined_subset[combined_subset$label == 0, feat, drop = TRUE]
  a <- combined_subset[combined_subset$label == 1, feat, drop = TRUE]
  
  boxplot(h, a,
          names = c("Human", "AI"),
          main  = feat,
          ylab  = feat,
          xlab  = "Group",
          col   = c("lightblue", "pink"))
}





# 2) Density plot: 여백 최소화 + 평균 기준선 + 색상 구분
par(mfrow = c(1, 3), mar = c(2, 2, 1.5, 1))

for (feat in features) {
  # 그룹별 벡터 추출
  h <- combined_subset[combined_subset$label == 0, feat, drop = TRUE]
  a <- combined_subset[combined_subset$label == 1, feat, drop = TRUE]
  
  # 밀도 계산
  d0 <- density(h, na.rm = TRUE)
  d1 <- density(a, na.rm = TRUE)
  
  # 빈 그래프
  plot(d0,
       main = feat,
       xlab = "", ylab = "",
       ylim = c(0, max(d0$y, d1$y)),
       col = "blue",      # H 밀도는 파란색
       lwd = 2)
  lines(d1, col = "red", lwd = 2)  # A 밀도는 빨간색
  
  # 평균값 기준선
  abline(v = mean(h, na.rm = TRUE), col = "blue", lty = 2)
  abline(v = mean(a, na.rm = TRUE), col = "red",  lty = 2)
  
  # 범례
  legend("topright",
         legend = c("Human", "AI"),
         col    = c("blue", "red"),
         lty    = c(1, 1),
         lwd    = c(2, 2),
         bty    = "n",
         cex    = 0.7)
}







# 3) ggplot2라는 고급 라이브러리 이용

library(ggplot2)
library(tidyr)

# 1) 데이터 통합 및 long-format으로 변환
combined_long <- pivot_longer(
  combined_subset,
  cols = -label,
  names_to = "feature",
  values_to = "value"
)

# 2) label을 문자열로 (Human / AI)
combined_long$label <- factor(combined_long$label, levels = c(0, 1), labels = c("Human", "AI"))

# 3) 각 feature별로 히스토그램 출력 -> 이 중에서, t-검정에서 가장 유의미하다고 판단되는 두가지 feature만을 ppt에 이용
unique_features <- unique(combined_long$feature)

# 4) 하나씩 출력
for (feat in unique_features) {
  cat("===== ", feat, " =====\n")
  
  p <- ggplot(subset(combined_long, feature == feat), aes(x = value, fill = label)) +
    geom_histogram(alpha = 0.5, bins = 40, position = "identity") +
    scale_fill_manual(values = c("blue", "red")) +
    labs(title = feat, x = feat, y = "Count", fill = "Label") +
    theme_minimal()
  
  print(p)
  readline(prompt = "[Enter] 키를 누르면 다음 그래프로 넘어감...")
}

