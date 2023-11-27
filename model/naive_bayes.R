library(mice) # library for impute
library(tidyverse)
library(dplyr)
library(dlookr)
library(caret) 

library(e1071)

load("dataset/dataset.RData")

# Predict from model and calculate its accuracy
predict_accuracy <- function(model, test_df) {
  predictions <- predict(model, test_df)
  plot(predictions)
  pred.matrix = table(test_df$output, predictions)
  print(confusionMatrix(pred.matrix))
}

# Naive Bayes
set.seed(seed)
nb_model <- naiveBayes(output ~ ., data = train_df)
predict_accuracy(nb_model, test_df) # 0.8
set.seed(seed)
nb_model <- naiveBayes(output ~ ., data = train_df_p)
predict_accuracy(nb_model, test_df_p) # 0.8