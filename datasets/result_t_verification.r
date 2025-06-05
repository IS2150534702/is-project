# 데이터 불러오기
df <- read.csv("../checkpoints/compare/result.csv")

# 열 이름 보기 (혹시 모르니 확인)
colnames(df)

# 열 이름 짧게 변수로 저장
main   <- df[[1]]
aux1   <- df[[2]]
aux2   <- df[[3]]
aux12  <- df[[4]]

# paired t-test 수행
cat("main vs aux1:\n")
print(t.test(main, aux1, paired = TRUE))

cat("\nmain vs aux2:\n")
print(t.test(main, aux2, paired = TRUE))

cat("\nmain vs aux1+aux2:\n")
print(t.test(main, aux12, paired = TRUE))

cat("\naux1 vs aux2:\n")
print(t.test(aux1, aux2, paired = TRUE))

cat("\naux1 vs aux1+aux2:\n")
print(t.test(aux1, aux12, paired = TRUE))

cat("\naux2 vs aux1+aux2:\n")
print(t.test(aux2, aux12, paired = TRUE))

