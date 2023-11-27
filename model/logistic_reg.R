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
  predictions <- ifelse(predictions > 0.5, 1, 0)
  pred.matrix = table(test_df$output, predictions)
  print(confusionMatrix(pred.matrix))
}

# Logistic Regression # New data pre-processing method needed
set.seed(seed)
log_model = glm(output ~ ., data = train_df, family = binomial)
predict_accuracy(log_model, test_df) # 0.8333