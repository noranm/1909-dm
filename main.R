##################################################
### 1711508 노 혜림
### Data Mining (2019-2)
### Department of Statistics
### Sookmyung Women's University
##################################################
#install.packages("Rtsne")
#install.packages("ggplot2")
#install.packages("rpart")
#install.packages("mice")
#install.packages("glmnet")
#install.packages("randomForest")
#install.packages("gbm")
#install.packages("pROC")
#install.packages("caret")
#install.packages("vip")
library(Rtsne) #자료요약
library(ggplot2) #시각화
library(rpart) #분류회귀나무
library(mice) #결측 다중대체
library(glmnet) #라쏘
library(randomForest) #rf
library(gbm) #gradient boost
library(pROC) #성능평가
library(caret) #성능평가
library(vip) # 변수 중요도

#########################################################
# DATA
#
heart.df <- read.csv('c:/datamining/heart.csv', header = TRUE)
## 1.2 정보 확인
dim(heart.df) # 1025 * 14
heart.df <- unique(heart.df) # 중복된 관측이 많아, 중복 관측을 제외
dim(heart.df) # 302 * 14
head(heart.df) 
str(heart.df)
## 결측확인 -> 결측 없음
colSums(is.na(heart.df))

# 변수 이름 보기 쉽게 재설정
colnames(heart.df) <- c("age", "sex", "chest_pain", "rest_blood_pressure", 
                        "cholestrol", "fasting_blood_sugar", "ST_wave", 
                        "max_heartrate", "Exercise_angina", "ST_depression", 
                        "slope", "n_major_vessels", "thalassemia", "target")
str(heart.df) #확인

########################## t-sne ###############################
#
#
set.seed(52)
tsne_model = Rtsne(as.matrix(heart.df), check_duplicates=FALSE,
                   pca = TRUE, perplexity = 30, theta = 0.5, dims=2)
d_tsne = as.data.frame(tsne_model$Y)
heart_tsne <- ggplot(d_tsne, aes(x=V1, y=V2, col = factor(heart.df$target))) 
heart_tsne + geom_point(size = 1) + scale_size_area(guide = FALSE)+ guides(colour = guide_legend(override.aes = list(size=6))) + xlab("") +ylab("") + ggtitle("T-SNE") +  theme(axis.text.x = element_blank(), axis.text.y = element_blank()) + scale_colour_brewer(palette = "Set2") + theme_light() +theme(legend.position = "top")

########################## 전처리 ###############################
#
#
# 양적(연속) 변수 heart_num
heart_num <- heart.df[,c(1, 4, 5, 8, 10, 14)]

# 기초 통계량
summary(heart_num)

# target에 따른 box plot
for (i in 1:5) {
  boxplot(heart_num[,i]~heart_num$target,
          main=colnames(heart_num)[i])
}

# 이상치 대체
heart.df$cholestrol[heart.df['cholestrol'] >= 400] <- (275 + (275-211) * 1.5) # 371
heart.df["max_heartrate"][heart.df['max_heartrate']<=80]<-(132+(132-149)*1.5) #106.5
heart.df["ST_depression"][heart.df["ST_depression"] >=5 ] <- (1.8 + (1.8-0) * 1.5) #4.5

# 범주&이산형 변수
heart_cat <- heart.df[,-c(1, 4, 5, 8, 10)]

## bar_plot
ggplot(heart_cat, aes(sex)) + geom_bar(aes(fill = factor(target))) +ylab("COUNT") + ggtitle("Sex") + theme_light()
ggplot(heart_cat, aes(chest_pain)) + geom_bar(aes(fill = factor(target))) + ylab("COUNT") + ggtitle("Chest Pain type") + theme_light()
ggplot(heart_cat, aes(fasting_blood_sugar)) + geom_bar(aes(fill = factor(target))) + ylab("COUNT") + ggtitle("fasting blood sugar(> 120 mg/dl)") + theme_light()
ggplot(heart_cat, aes(ST_wave)) + geom_bar(aes(fill = factor(target))) + ylab("COUNT") + ggtitle("ST-T wave type") + theme_light()
ggplot(heart_cat, aes(Exercise_angina)) + geom_bar(aes(fill = factor(target))) + ylab("COUNT") + ggtitle("Exercise induced angina") + theme_light()
ggplot(heart_cat, aes(slope)) + geom_bar(aes(fill = factor(target))) + ylab("COUNT") + ggtitle("Slope") + theme_light()
ggplot(heart_cat, aes(n_major_vessels)) + geom_bar(aes(fill = factor(target))) + ylab("COUNT") + ggtitle("# of major vessels") + theme_light()
ggplot(heart_cat, aes(thalassemia)) + geom_bar(aes(fill = factor(target))) + ylab("COUNT") + ggtitle("Thalassemia") + theme_light()

# 적은 수의 범주 대체
heart.df["n_major_vessels"][heart.df["n_major_vessels"] == 4] <- NA #4개 0.0132
heart.df["thalassemia"][heart.df["thalassemia"] == 0] <- NA #2개 0.0066
heart.df["ST_wave"][heart.df["ST_wave"] == 2] <- NA #4개

# 범주형 변수 통합 (exercise_angina & chest_pain )
angina.table <- table("exercise" = heart.df$Exercise_angina, 
                      "chest pain" = heart.df$chest_pain, 
                      "target" = heart.df$target)

# 범주의 빈도 확인 (Freq)
total.angina <- data.frame(angina.table[,,1]+angina.table[,,2])
# 버블차트 (4/302=0.01324503, 11/302=0.03642384, 4/302=0.01324503)
ggplot(total.angina, aes(x=chest.pain , y = exercise)) + geom_point(aes(size = Freq), shape=21, colour = "white", fill = "#3399FF") + scale_size_area(max_size = 20, guide = FALSE) +  geom_text(aes(y = as.numeric(exercise) - sqrt(Freq)/30, label = Freq),  color = "black", size = 4) + theme_light()

# angina.table[,,2] 는 target이 1인 값
# percentage = target이 1인 값 / (target 1인 값 + target 0인 값)
total.angina$per <- data.frame(angina.table[,,2])$Freq/total.angina$Freq

# barplot(y : percentage, x: chest_pain 과 exercise_angina 각각 8개의 범주)
ggplot(total.angina, aes(x=chest.pain, y=per, fill= factor(exercise))) + geom_bar(stat = "identity", position =position_dodge(0.5), width = 0.5) + geom_text(aes(label = sprintf("%2.1f %%",per*100)), vjust = -0.4)

# chest_pain's 0 -> 0과 0.5(exercsie_angina = 1)으로 임의 구분
rule <- (heart.df$chest_pain == 0) & (heart.df$Exercise_angina == 1) 
heart.df$chest_pain[rule] <-0.5
table(heart.df$chest_pain) #확인 143 = 63+80

# Exercise_angina 제외
heart.df <- heart.df[,-9]
ggplot(heart.df, aes(x= factor(chest_pain))) + geom_bar(aes(fill=factor(target)), width = 0.4) + theme(axis.text.x = element_text(vjust=0.6))

# 가변수 처리
# (1) chest pain 
heart.df$chest_pain <- factor(heart.df[,"chest_pain"], levels = c(0, 0.5, 1, 2, 3),
                              labels = c("typical", "typical_exercise", "atypical", "non-pain", "asymtomatic"))
# (2) slope
heart.df$slope <- factor(heart.df[,"slope"], levels = c(0, 1, 2),
                         labels = c("upslopping", "flat", "downslopping"))

# (3) thalassemia 유전적 결함이 심한 정도가 1<3<2 이기에 순서로 생각
# thalassemia > 1: normal, 2: reversable defect, 3: fixed defect 값 변경
table(heart.df$thalassemia) #확인 18, 165, 117
heart.df$thalassemia[heart.df['thalassemia'] == 2] <- 2.5
heart.df$thalassemia[heart.df['thalassemia'] == 3] <- 2
heart.df$thalassemia[heart.df['thalassemia'] == 2.5] <- 3
table(heart.df$thalassemia) #확인 18, 117, 165
# (4) sex, fasting_blood_sugar, ST_wave 는 0,1 의 범주이기에 가변수 생성X
# (5) major_vessels 이산형

str(heart.df) #check
# corr
heatmap(cor(na.omit(heart.df[,-c(3,10,13)])), Rowv=NA, Colv=NA)
########################### missing value ##############################
#
#
heart.rpart <- heart.df # rpart로 결측 대체한 df
heart2 <- heart.df[,-13] # 결측 대체를 위한 target이 없는 df
colSums(is.na(heart2)) # ST_wave 4, major_vessels 4, thalassemia 2
md.pattern(heart2)

###### [1] rpart ######
## (1) ST_wave~ 
table(heart2$ST_wave) # 0: 147, 1:151

## y = ST_wave 
ST_modi1 <- rpart(ST_wave ~ ., heart2, method = "class", na.action = na.omit)
ST_pred1 <- predict(ST_modi1, heart2[is.na(heart2$ST_wave),])[,1] # p(normal) 

## ST_pred$normal < 0.5 이면 abnormal 비정상, normal(0) abnormal(1)
heart.rpart[is.na(heart.rpart$ST_wave),"ST_wave"] <- 1*(ST_pred1<0.5)

table(heart.rpart$ST_wave) # 0: 147+3=150, 1: 151+1=152 
sum(is.na(heart.rpart$ST_wave)) # 확인 결측없음

## (2) major vessels ~ 18
table(heart2$n_major_vessels) #0:175, 1:65, 2:38, 3:20

## y = n_major_vessels
vessels_modi1 <- rpart(n_major_vessels ~ .,heart2, 
                       method="class", na.action = na.omit)
vessels_pred1 <- predict(vessels_modi1,
                         heart2[is.na(heart2$n_major_vessels),])
vessels_pred1 <- data.frame(vessels_pred1)

head(vessels_pred1) # "X0" : p(0), "X1" : p(1), "X2" : p(2), "X3" : p(3)
n_vessels <- nrow(vessels_pred1)

class_v1 <- rep(1, each=n_vessels)
for (i in 1:n_vessels) {
  max = 0
  if( max < vessels_pred1[i,"X0"]){
    class = 0
    max = vessels_pred1[i,"X0"] 
  }
  if ( max < vessels_pred1[i,"X1"]){
    class = 1
    max = vessels_pred1[i,"X1"]
  }
  if ( max < vessels_pred1[i,"X2"]){
    class = 2
    max = vessels_pred1[i,"X2"]
  }
  if ( max < vessels_pred1[i,"X3"]){
    class = 3
    max = vessels_pred1[i,"X3"] 
  }
  class_v1[i] <- class
}

heart.rpart[is.na(heart.rpart$n_major_vessels),"n_major_vessels"] <- class_v1

table(heart.rpart$n_major_vessels) #0:175+4=179, 1:65, 2:38, 3:20
sum(is.na(heart.rpart$n_major_vessels)) # 확인 결측없음

## (3) thalassemia ~ 2
sum(is.na(heart2$thalassemia))

thal_modi1 <- rpart(thalassemia ~ .,heart2, 
                    method="class", na.action = na.omit)
thal_pred1 <- predict(thal_modi1,
                      heart2[is.na(heart2$thalassemia),])
thal_pred1 <- data.frame(thal_pred1)
thal_pred1

n_thal <- nrow(thal_pred1)
class_t1 <- rep(1, each=n_thal) # 확인용

for (i in 1:n_thal) {
  max = 0
  if( max < thal_pred1[i,"X1"]){
    class = 1
    max = thal_pred1[i,"X1"] 
  }
  if ( max < thal_pred1[i,"X2"]){
    class = 2
    max = thal_pred1[i,"X2"]
  }
  if (max < thal_pred1[i,"X3"]) {
    class = 3
    max = thal_pred1[i,"X3"]
  }
  class_t1[i] <- class
}
heart.rpart[is.na(heart.rpart$thalassemia),"thalassemia"] <- class_t1

table(heart.rpart$thalassemia) #0:18, 2:117+1=118, 3:165+1=166
sum(is.na(heart.rpart$thalassemia)) # 확인 결측없음

###### [2] MICE ######
heart.mice <- heart.df # mice로 대체할 df
colSums(is.na(heart.mice))
# colSums(is.na(heart2)) # (앞에서 사용한 target없는 데이터셋 heart2)

# [meth]옵션 ; classfication and regression tree
mice_mod <- mice(heart2, maxit=10, meth = "cart", seed=52)

## 값 확인
mice_output <- complete(mice_mod, 3)[,c("ST_wave", "n_major_vessels", "thalassemia")]

table(mice_output$ST_wave) # 0:147+3=150, 1:151+1=152 
table(mice_output$n_major_vessels) # 0:175+2=177, 1:65+1=66, 2:38+1=39, 3:20
table(mice_output$thalassemia) # 1:18, 2:117+2=119, 3:165

heart.mice[,"ST_wave"] <- mice_output$ST_wave
heart.mice[,"n_major_vessels"] <- mice_output$n_major_vessels
heart.mice[,"thalassemia"] <- mice_output$thalassemia

colSums(is.na(heart.mice))

#########################################################
# METHOD : set.seed 꼭 하기 
#
n <- nrow(heart.df)

set.seed(52) 
i_perm <-  sample.int(n)
numFolds <- 5                                           

########################## Fit Model ###############################
#
### glmnet's setting
glmnet.lambda <- 2^seq(from=1, to=-10, length=100)

resTemp1 <- NULL
rm.glmnet_table <- NULL
rpart.glmnet_table <- NULL
mice.glmnet_table <- NULL

### rf's
max_tree = 100
n_tree_seq = seq(from=10, to=max_tree, by=10)

resTemp3 <- NULL
rm.rf_table <- NULL
rpart.rf_table <- NULL
mice.rf_table <- NULL

#### gb's setting
hpTable_gb = expand.grid(
  max_depth = c(1, 2, 3, 4), 
  shrnk = seq(0.05, 0.25, 0.0005))

# max_tree, n_tree_seq 는 랜덤포레스트와 함께 사용
resTemp2 <- NULL
rm.gb_table <- NULL
rpart.gb_table <- NULL
mice.gb_table <- NULL

######################################################
# (1) heart.rm 
for (k in 1:numFolds){
  # 열 구분
  ind_start <- floor((k-1)/numFolds*n)+1
  ind_end <- floor(k/numFolds*n)
  ind_val <- i_perm[ind_start:ind_end]
  # 분할 
  df_valid <- na.omit(heart.df[ind_val,])
  df_train <- na.omit(heart.df[setdiff(1:n, ind_val),])
  
  ######################### Glmnet #########################
  rm.glmnet <- glmnet( x = as.matrix(model.matrix(~ .,df_train[,-13])[,-1]),
                       y = as.numeric(df_train[,13]),
                       family = "binomial",
                       lambda=glmnet.lambda)
  # predict
  rm.glmnet_pred <- predict(rm.glmnet, 
                            as.matrix(model.matrix(~.,df_valid[,-13])[,-1]),
                            type="response")
  # error 계산
  for (i in 1:100) {
    rm.glmnet_auc <- pROC::auc(as.vector(df_valid$target),
                               as.vector(rm.glmnet_pred[,i]), 
                               levels=c(0,1), direction="<") 
    err_temp <- 1-rm.glmnet_auc
    resTemp1 <- data.frame(k_fold = k,
                           lambda = glmnet.lambda[i],
                           error = err_temp)
    rm.glmnet_table <- rbind(rm.glmnet_table, resTemp1)
  }
  
  
  ######################### RandomForest #########################
  for (i in 1:length(n_tree_seq)) {
    rm.rf <- randomForest(as.factor(target) ~ ., 
                          data=df_train, ntree=n_tree_seq[i], proximity=T)
    # predict
    rm.rf_pred <- predict(rm.rf, df_valid[,-13],
                          type="prob")
    
    # error 계산
    rm.rf_auc <- pROC::auc(df_valid$target, rm.rf_pred[,"1"],
                           levels=c(0,1), direction="<")
    err_temp <- 1-rm.rf_auc
    resTemp2 <- data.frame(k_fold = k,
                           n_tree = n_tree_seq[i],
                           error = err_temp)
    rm.rf_table = rbind(rm.rf_table, resTemp2)
  }
  
  ######################### Gradient Boost #########################
  for (i in 1:nrow(hpTable_gb)){
    rm.gbm <- gbm(target~., data = df_train, 
                  distribution = "bernoulli",
                  n.trees=max_tree,
                  shrinkage = hpTable_gb$shrnk[i],
                  interaction.depth=hpTable_gb$max_depth[i],
                  n.minobsinnode = 10)
  
    rm.gbm_preds <- predict(rm.gbm, 
                            newdata = as.data.frame(df_valid[,-13]), 
                            type = "response", n.tree=n_tree_seq)
    
    for (j in 1:length(n_tree_seq)){
      n.tree = n_tree_seq[j]
      rm.gbm_pred <- rm.gbm_preds[,j]
      rm.gbm_auc <- pROC::auc(df_valid$target, rm.gbm_pred,
                              levels=c(0,1), direction="<")
      
      resTemp3 <- data.frame(k_fold = k,
                            max_depth = hpTable_gb$max_depth[i],
                            shrnk = hpTable_gb$shrnk[i],
                            n.tree = n.tree,
                            error = 1-rm.gbm_auc)
      rm.gb_table = rbind(rm.gb_table, resTemp3)
    }
  }
}
#### k ~ error 평균 
## glmnet 
rm.glmnet_table2 <- aggregate(error~lambda,
                              rm.glmnet_table, mean)
print(rm.glmnet_table2[order(rm.glmnet_table2$error),][1,]) # check
rm.glmnet_err <- rm.glmnet_table2[order(rm.glmnet_table2$error),][1,] 
#28 lambda = 0.0078125, err = 0.09789467, ok

## rf
rm.rf_table2 <- aggregate(error~n_tree,
                          rm.rf_table, mean)
print(rm.rf_table2[order(rm.rf_table2$error),][1,])
rm.rf_err <- rm.rf_table2[order(rm.rf_table2$error),][1,] 
#70 n_tree = 70, error = 0.1042622, ok

## gb
rm.gb_table2 <- aggregate(error~max_depth+shrnk+n.tree, rm.gb_table, mean)
print(rm.gb_table2[order(rm.gb_table2$error),][1,])
rm.gb_err <- rm.gb_table2[order(rm.gb_table2$error),][1,] 
#9213 max_depth = 1 shrink = 0.199, n.tree = 60, error = 0.08217781, ok

################################################
# (2) heart.rpart & heart.mice


resTemp1 <- NULL
resTemp2 <- NULL
resTemp3 <- NULL

for (k in 1:numFolds){
  # 열 구분
  ind_start <- floor((k-1)/numFolds*n)+1
  ind_end <- floor(k/numFolds*n)
  ind_val <- i_perm[ind_start:ind_end]
  # 분할 (1) rpart
  df_valid1 <- heart.rpart[ind_val,]
  df_train1 <- heart.rpart[setdiff(1:n, ind_val),]
  # 분할 (2) mice
  df_valid2 <- heart.mice[ind_val,]
  df_train2 <- heart.mice[setdiff(1:n, ind_val),]
  
  ######################### Glmnet #########################
  # (1) rpart
  rpart.glmnet <- glmnet( x = as.matrix(model.matrix(~ .,df_train1[,-13])[,-1]),
                       y = as.numeric(df_train1[,13]),
                       family = "binomial",
                       lambda = glmnet.lambda)
  # predict
  rpart.glmnet_pred <- predict(rpart.glmnet, 
                            as.matrix(model.matrix(~.,df_valid1[,-13])[,-1]),
                            type="response")
  # error 계산
  for (i in 1:100) {
    rpart.glmnet_auc <- pROC::auc(as.vector(df_valid1$target),
                               as.vector(rpart.glmnet_pred[,i]), 
                               levels=c(0,1), direction="<") 
    err_temp <- 1-rpart.glmnet_auc
    resTemp1 <- data.frame(k_fold = k,
                           lambda = glmnet.lambda[i],
                           error = err_temp)
    rpart.glmnet_table <- rbind(rpart.glmnet_table, resTemp1)
  }
  
  # (2) mice
  mice.glmnet <- glmnet( x = as.matrix(model.matrix(~ .,df_train2[,-13])[,-1]),
                          y = as.numeric(df_train2[,13]),
                          family = "binomial",
                         lambda = glmnet.lambda)
  # predict
  mice.glmnet_pred <- predict(mice.glmnet, 
                               as.matrix(model.matrix(~.,df_valid2[,-13])[,-1]),
                               type="response")
  # error 계산
  for (i in 1:100) {
    mice.glmnet_auc <- pROC::auc(as.vector(df_valid2$target),
                                  as.vector(mice.glmnet_pred[,i]), 
                                  levels=c(0,1), direction="<") 
    err_temp <- 1-mice.glmnet_auc
    resTemp1 <- data.frame(k_fold = k,
                           lambda = glmnet.lambda[i],
                           error = err_temp)
    mice.glmnet_table <- rbind(mice.glmnet_table, resTemp1)
  }
  
  ######################### RandomForest #########################
  for (i in 1:length(n_tree_seq)) {
    # (1) rpart
    rpart.rf <- randomForest(as.factor(target) ~ ., 
                          data=df_train1, ntree=n_tree_seq[i], proximity=T)
    # predict
    rpart.rf_pred <- predict(rpart.rf, df_valid1[,-13], type="prob")
    
    # error 계산
    rpart.rf_auc <- pROC::auc(df_valid1$target, rpart.rf_pred[,"1"],
                           levels=c(0,1), direction="<")
    
    err_temp <- 1-rpart.rf_auc
    resTemp2 <- data.frame(k_fold = k,
                           n_tree = n_tree_seq[i],
                           error = err_temp)
    rpart.rf_table = rbind(rpart.rf_table, resTemp2)
    
    # (2) mice
    mice.rf <- randomForest(as.factor(target) ~ ., 
                            data=df_train2, ntree=n_tree_seq[i], proximity=T)
    # predict
    mice.rf_pred <- predict(mice.rf, df_valid2[,-13],
                            type="prob")
    
    # error 계산
    mice.rf_auc <- pROC::auc(df_valid2$target, mice.rf_pred[,"1"],
                             levels=c(0,1), direction="<")
    err_temp <- 1-mice.rf_auc
    resTemp2 <- data.frame(k_fold = k,
                           n_tree = n_tree_seq[i],
                           error = err_temp)
    mice.rf_table = rbind(mice.rf_table, resTemp2)    
  }
  
  ######################### Gradient Boost #########################
  # (1) rpart
  for (i in 1:nrow(hpTable_gb)){
    rpart.gbm <- gbm(target~., data = df_train1, 
                  distribution = "bernoulli",
                  n.trees=max_tree,
                  shrinkage = hpTable_gb$shrnk[i],
                  interaction.depth=hpTable_gb$max_depth[i],
                  n.minobsinnode = 10)
    
    rpart.gbm_preds <- predict(rpart.gbm, 
                            newdata = as.data.frame(df_valid1[,-13]), 
                            type = "response", n.tree=n_tree_seq)
    
    for (j in 1:length(n_tree_seq)){
      n.tree = n_tree_seq[j]
      rpart.gbm_pred <- rpart.gbm_preds[,j]
      rpart.gbm_auc <- pROC::auc(df_valid1$target, rpart.gbm_pred,
                              levels=c(0,1), direction="<")
      
      resTemp3 <- data.frame(k_fold = k,
                             max_depth = hpTable_gb$max_depth[i],
                             shrnk = hpTable_gb$shrnk[i],
                             n.tree = n.tree,
                             error = 1-rpart.gbm_auc)
      rpart.gb_table = rbind(rpart.gb_table, resTemp3)
    }
  }
  # (2) mice
  for (i in 1:nrow(hpTable_gb)){
    mice.gbm <- gbm(target~., data = df_train2, 
                     distribution = "bernoulli",
                     n.trees=max_tree,
                     shrinkage = hpTable_gb$shrnk[i],
                     interaction.depth=hpTable_gb$max_depth[i],
                     n.minobsinnode = 10)
    
    mice.gbm_preds <- predict(mice.gbm, 
                               newdata = as.data.frame(df_valid2[,-13]), 
                               type = "response", n.tree=n_tree_seq)
    
    for (j in 1:length(n_tree_seq)){
      n.tree = n_tree_seq[j]
      mice.gbm_pred <- mice.gbm_preds[,j]
      mice.gbm_auc <- pROC::auc(df_valid2$target, mice.gbm_pred,
                                 levels=c(0,1), direction="<")
      
      resTemp2 <- data.frame(k_folds = k,
                             max_depth = hpTable_gb$max_depth[i],
                             shrnk = hpTable_gb$shrnk[i],
                             n.tree = n.tree,
                             error = 1-mice.gbm_auc)
      mice.gb_table = rbind(mice.gb_table, resTemp2)
    }
  }
}

# k ~ error 평균 
## glmnet
### (1) rpart
rpart.glmnet_table2 <- aggregate(error~lambda,
                              rpart.glmnet_table, mean)
print(rpart.glmnet_table2[order(rpart.glmnet_table2$error),][1,]) # check
rpart.glmnet_err <- rpart.glmnet_table2[order(rpart.glmnet_table2$error),][1,] 
# 18 0.003616698 0.09429403, ok

### (2) mice
mice.glmnet_table2 <- aggregate(error~lambda,
                                 mice.glmnet_table, mean)
print(mice.glmnet_table2[order(mice.glmnet_table2$error),][1,]) # check
mice.glmnet_err <- mice.glmnet_table2[order(mice.glmnet_table2$error),][1,]
# 16 0.003100393 0.09474763, ok

## rf
### (1) rpart
rpart.rf_table2 <- aggregate(error~n_tree, rpart.rf_table, mean)
print(rpart.rf_table2[order(rpart.rf_table2$error),][1,])
rpart.rf_err <- rm.rf_table2[order(rm.rf_table2$error),][1,] 
#9 90 0.0988104, ok

### (2) mice
mice.rf_table2 <- aggregate(error~n_tree, mice.rf_table, mean)
print(mice.rf_table2[order(mice.rf_table2$error),][1,])
mice.rf_err <- mice.rf_table2[order(mice.rf_table2$error),][1,] 
#9 90 0.09976039, ok

## gbm
### (1) rpart
rpart.gb_table2 <- aggregate(error~max_depth+shrnk+n.tree, rpart.gb_table, mean)
print(rpart.gb_table2[order(rpart.gb_table2$error),][1,])
rpart.gb_err <- rpart.gb_table2[order(rpart.gb_table2$error),][1,] 
#3036 4 0.2285 20 0.07840075, ok

### (2) mice
mice.gb_table2 <- aggregate(error~max_depth+shrnk+n.tree, mice.gb_table, mean)
print(mice.gb_table2[order(mice.gb_table2$error),][1,])
mice.gb_err <- mice.gb_table2[order(mice.gb_table2$error),][1,] 
#12301 1 0.1840 80 0.07761351, ok

#### 비교하기 ####
dataset <- rep(c("na.removed","rpart", "mice"), times=3)
method <- rep(c("glmnet","RandomForest","Gradient Boost"), each=3)
error = c(rm.glmnet_err$error, rpart.glmnet_err$error, mice.glmnet_err$error,
          rm.rf_err$error, rpart.rf_err$error, mice.rf_err$error,
          rm.gb_err$error, rpart.gb_err$error, mice.gb_err$error)
cm_err <- data.frame(dataset, method, error)

# error 오름차순 정렬 
cm_err[order(cm_err$error),]

# glmnet 결과
heart.rm <- na.omit(heart.df)
glmnet_result <- rbind(na.removed = rm.glmnet_err, rpart = rpart.glmnet_err, mice = mice.glmnet_err)

rm.glmnet_result <- glmnet(x=as.matrix(model.matrix(~ ., heart.rm[,-13])[,-1]),
                           y = as.numeric(heart.rm[,13]), 
                           family = "binomial",
                           lambda = rm.glmnet_err$lambda)
rm.glmnet_beta <- as.data.frame(as.matrix(coef(rm.glmnet_result)))
rm.glmnet_beta$name <- row.names(rm.glmnet_beta)

rpart.glmnet_result <- glmnet(x=as.matrix(model.matrix(~ ., heart.rpart[,-13])[,-1]),
                           y = as.numeric(heart.rpart[,13]), 
                           family = "binomial",
                           lambda = rpart.glmnet_err$lambda)
rpart.glmnet_beta <- as.data.frame(as.matrix(coef(rpart.glmnet_result)))
rpart.glmnet_beta$name <- row.names(rpart.glmnet_beta)

mice.glmnet_result <- glmnet(x=as.matrix(model.matrix(~ ., heart.mice[,-13])[,-1]),
                              y = as.numeric(heart.mice[,13]), 
                              family = "binomial",
                              lambda = mice.glmnet_err$lambda)
mice.glmnet_beta <- as.data.frame(as.matrix(coef(mice.glmnet_result)))
mice.glmnet_beta$name <- row.names(mice.glmnet_beta)

# 베타 값 시각화
# intercept 제외하고 베타 크기에 따른 그래프.
ggplot(rm.glmnet_beta[-1,], aes(x = s0, y = reorder(name, s0))) + geom_point(size = 3, color = ifelse((abs(rm.glmnet_beta$s0[-1])>=0.1),"red","grey60")) + theme_bw()  + theme(panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(), panel.grid.major.y = element_line(color = "grey60", linetype = "dashed")) + geom_vline(xintercept=0, linetype=2, color='black') + ylab(" ") + xlab("beta")+ggtitle("na_removed model") 
ggplot(rpart.glmnet_beta[-1,], aes(x = s0, y = reorder(name, s0))) + geom_point(size = 3, color = ifelse((abs(rpart.glmnet_beta$s0[-1])>=0.1),"red","grey60")) + theme_bw()  + theme(panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(), panel.grid.major.y = element_line(color = "grey60", linetype = "dashed")) + geom_vline(xintercept=0, linetype=2, color='black') + ylab(" ") + xlab("beta")+ggtitle("rpart model")
ggplot(mice.glmnet_beta[-1,], aes(x = s0, y = reorder(name, s0))) + geom_point(size = 3, color = ifelse((abs(mice.glmnet_beta$s0[-1])>=0.1),"red","grey60")) + theme_bw()  + theme(panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(), panel.grid.major.y = element_line(color = "grey60", linetype = "dashed")) + geom_vline(xintercept=0, linetype=2, color='black') + ylab(" ") + xlab("beta")+ggtitle("mice model")

# randomForest 결과
rf_result <- rbind(na.removed = rm.rf_err, rpart = rpart.rf_err, mice = mice.rf_err)
rm.rf_result <- randomForest(as.factor(target) ~ ., 
                         data=heart.rm, ntree=rm.rf_err$n_tree, proximity=T)
rm.rf_impt <- data.frame(name = colnames(heart2), rm.rf_result$importance)

rpart.rf_result <- randomForest(as.factor(target) ~ ., 
                                data=heart.rpart, ntree=rpart.rf_err$n_tree, proximity=T)
rpart.rf_impt <- data.frame(name = colnames(heart2), rpart.rf_result$importance)

mice.rf_result <- randomForest(as.factor(target) ~ ., 
                               data=heart.mice, ntree=mice.rf_err$n_tree, proximity=T)
mice.rf_impt <- data.frame(name = colnames(heart2), mice.rf_result$importance)

# variable importance 시각화
ggplot(rm.rf_impt, aes(x = MeanDecreaseGini, y = reorder(name, MeanDecreaseGini))) + geom_point(size = 3, color = ifelse(rm.rf_impt$MeanDecreaseGini>10,"red","grey60")) + theme_bw()  + theme(panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(), panel.grid.major.y = element_line(color = "grey60", linetype = "dashed")) + xlab("variable importance") + ylab(" ") + ggtitle("na removed model")
ggplot(rpart.rf_impt, aes(x = MeanDecreaseGini, y = reorder(name, MeanDecreaseGini))) + geom_point(size = 3, color = ifelse(rpart.rf_impt$MeanDecreaseGini>10,"red","grey60")) + theme_bw()  + theme(panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(), panel.grid.major.y = element_line(color = "grey60", linetype = "dashed")) + xlab("variable importance") + ylab(" ") + ggtitle("rpart model")
ggplot(mice.rf_impt, aes(x = MeanDecreaseGini, y = reorder(name, MeanDecreaseGini))) + geom_point(size = 3, color = ifelse(mice.rf_impt$MeanDecreaseGini>10,"red","grey60")) + theme_bw()  + theme(panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(), panel.grid.major.y = element_line(color = "grey60", linetype = "dashed")) + xlab("variable importance") + ylab(" ") + ggtitle("mice model")

# Gradient Boost 결과
gbm_result <- rbind(na.removed = rm.gb_err, rpart = rpart.gb_err, mice = mice.gb_err)
rm.gbm_result <- gbm(target~., data = heart.rm, 
                     distribution = "bernoulli",
                     n.trees=rm.gb_err$n.tree,
                     shrinkage = rm.gb_err$shrnk,
                     interaction.depth=rm.gb_err$max_depth,
                     n.minobsinnode = 10)
rpart.gbm_result <- gbm(target~., data = heart.rpart, 
                     distribution = "bernoulli",
                     n.trees=rpart.gb_err$n.tree,
                     shrinkage = rpart.gb_err$shrnk,
                     interaction.depth=rpart.gb_err$max_depth,
                     n.minobsinnode = 10)
mice.gbm_result <- gbm(target~., data = heart.mice, 
                     distribution = "bernoulli",
                     n.trees=mice.gb_err$n.tree,
                     shrinkage = mice.gb_err$shrnk,
                     interaction.depth=mice.gb_err$max_depth,
                     n.minobsinnode = 10)

#gbm, variable importance
rm.gbm_imp <- vip(rm.gbm_result)
vip(rm.gbm_result, bar = FALSE, color = ifelse(rm.gbm_imp$data$Importance>5,"red","grey60"), shape = , size = 4) +theme_light()+ theme(panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(), panel.grid.major.y = element_line(color = "grey60", linetype = "dashed")) + xlab("variable importance") + ylab(" ") + ggtitle("na removed model")
rpart.gbm_imp <- vip(rpart.gbm_result)
vip(rpart.gbm_result, bar = FALSE, color = ifelse(rpart.gbm_imp$data$Importance>5,"red","grey60"), shape = , size = 4) +theme_light()+ theme(panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(), panel.grid.major.y = element_line(color = "grey60", linetype = "dashed")) + xlab("variable importance") + ylab(" ") +ggtitle("rpart model")
mice.gbm_imp <- vip(mice.gbm_result)
vip(mice.gbm_result, bar = FALSE, color = ifelse(mice.gbm_imp$data$Importance>5,"red","grey60"), shape = , size = 4) +theme_light()+ theme(panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(), panel.grid.major.y = element_line(color = "grey60", linetype = "dashed")) + xlab("variable importance") + ylab(" ") + ggtitle("mice model")

# result.Rdata codes에 함께 첨부됨 **주소확인** 
save(rm.glmnet_result, rpart.glmnet_result, mice.glmnet_result,
     rm.rf_result, rpart.rf_result, mice.rf_result,
     rm.gbm_result, rpart.gbm_result, mice.gbm_result, 
     file = "C:/datamining/result_model.Rdata")
