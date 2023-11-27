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
nn_model(train_df, test_df, c(13, 7, 3)) 
set.seed(seed)
nn_model(train_df, test_df, c(13, 7)) 
set.seed(seed)
nn_model(train_df, test_df, c(7, 3)) 
set.seed(seed)
nn_model(train_df, test_df, c(7)) 

df_p <- df
# Standardize numeric features
numeric_features <- c("age", "trtbps", "chol", "thalachh", "oldpeak")
df_p[numeric_features] <- scale(df_p[numeric_features])

# Create training and testing dataset
train_df_scaled <- df_p[r_train,]
test_df_scaled <- df_p[-r_train,]

# Neural Network with scaled data
set.seed(seed)
nn_model(train_df_scaled, test_df_scaled, c(13, 7, 3)) 
set.seed(seed)
nn_model(train_df_scaled, test_df_scaled, c(13, 7)) 
set.seed(seed)
nn_model(train_df_scaled, test_df_scaled, c(7, 3)) 
set.seed(seed)
nn_model(train_df_scaled, test_df_scaled, c(7)) 

