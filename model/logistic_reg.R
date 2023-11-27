library(mice) # library for impute
library(tidyverse)
library(dplyr)
library(dlookr)
library(caret) 

load("dataset/dataset.RData")

# Prediction on test data set from model and calculate model's accuracy
predict_accuracy <- function(model, test_df) {
  predictions <- predict(model, test_df)
  plot(predictions)
  pred.matrix = table(test_df$output, predictions)
  print(confusionMatrix(pred.matrix))
}

# Logistic Regression # New data pre-processing method needed
set.seed(seed)
log_model = glm(output ~ ., data = train_df, family = binomial)
predict_accuracy(log_model, test_df) # 0.8444, 0.8222

train_features_m <- as.matrix(train_df[, -which(names(train_df) == "output")])
test_features_m <- as.matrix(test_df[, -which(names(test_df) == "output")])

train_features_pm <- as.matrix(train_df_p[, -which(names(train_df_p) == "output")])
test_features_pm <- as.matrix(test_df_p[, -which(names(test_df_p) == "output")])