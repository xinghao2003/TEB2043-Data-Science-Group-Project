library(mice) # library for impute
library(tidyverse)
library(dplyr)
library(dlookr)
library(caret) 

library(class)

load("dataset/dataset.RData")

# Predict from model and calculate its accuracy
predict_accuracy <- function(model, test_df) {
  predictions <- predict(model, test_df)
  plot(predictions)
  pred.matrix = table(test_df$output, predictions)
  print(confusionMatrix(pred.matrix))
}

# KNN
# Training the KNN model
k <- 5 # Choosing the number of neighbors
set.seed(seed)
knn_model <- knn(train = train_features, test = test_features, cl = train_target, k = k)
# Calculating accuracy
accuracy <- mean(knn_model == test_target)
cat("Accuracy:", accuracy) # 0.6
set.seed(seed)
knn_model <- knn(train = train_features_p, test = test_features_p, cl = train_target_p, k = k)
# Calculating accuracy
accuracy <- mean(knn_model == test_target_p)
cat("Accuracy:", accuracy) # 0.8333