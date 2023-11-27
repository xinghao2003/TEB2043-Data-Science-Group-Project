# Load necessary libraries
library(mice)
library(tidyverse)
library(dplyr)
library(dlookr)
library(caret)
library(neuralnet)

# Load dataset
load("dataset/dataset.RData")

# Predict from model and calculate its accuracy
predict_accuracy <- function(model, test_df) {
  predictions <- predict(model, test_df)
  
  predictions <- ifelse(predictions > 0.5, 1, 0)
  
  # Print confusion matrix
  print(table(test_df$output, predictions))
  
  # Generate confusion matrix
  pred.matrix <- table(test_df$output, predictions)
  
  # Print additional information
  print(confusionMatrix(pred.matrix))
}

# Neural Network Model function
nn_model <- function(train_df, test_df, layer, linear = FALSE) {
  model <- neuralnet(
    output ~ .,
    data = train_df,
    hidden = layer,
    linear.output = linear
  )
  plot(model, rep = "best")
  predict_accuracy(model, test_df)
}

# Neural Network
set.seed(seed)
nn_model(train_df, test_df, c(13, 7, 3)) # 0.7033,0.6923
set.seed(seed)
nn_model(train_df, test_df, c(13, 7)) # 0.6154, 0.7033
set.seed(seed)
nn_model(train_df, test_df, c(7, 3)) # 0.7692, 0.7363
set.seed(seed)
nn_model(train_df, test_df, c(7)) # 0.6593, 0.6703

# Example: Scaling numeric features
numeric_features <- c("age", "trtbps", "chol", "thalachh", "oldpeak")
heart_scaled <- heart
heart_scaled[numeric_features] <- scale(heart_scaled[numeric_features])

# Example: Create 'train_df_scaled'
set.seed(123)
train_indices <- createDataPartition(heart_scaled$output, p = 0.7, list = FALSE)
train_df_scaled <- heart_scaled[train_indices, ]
test_df_scaled <- heart_scaled[-train_indices, ]

# Neural Network with scaled data excluded "output"
set.seed(seed)
nn_model(train_df_scaled, test_df_scaled, c(13, 7, 3)) # 0.7912, 0.7802
set.seed(seed)
nn_model(train_df_scaled, test_df_scaled, c(13, 7)) # 0.8132, 0.8022
set.seed(seed)
nn_model(train_df_scaled, test_df_scaled, c(7, 3)) # 0.7692, 0.7582
set.seed(seed)
nn_model(train_df_scaled, test_df_scaled, c(7)) # 0.7912, 0.8043