library(mice) # library for impute
library(tidyverse)
library(dplyr)
library(dlookr)
library(caret) 

library(neuralnet)

load("dataset/dataset.RData")

# Predict from model and calculate its accuracy
predict_accuracy <- function(model, test_df) {
  predictions <- predict(model, test_df)
  plot(predictions)
  pred.matrix = table(test_df$output, predictions)
  print(confusionMatrix(pred.matrix))
}

# Neural Network Model function
nn_model <- function(train_df, test_df, layer, linear = FALSE) {
  model = neuralnet(
    output~.,
    data=train_df,
    hidden=layer,
    linear.output = linear
  )
  plot(model,rep = "best")
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
# Neural Network with scaled data excluded "output"
set.seed(seed)
nn_model(train_df_scaled, test_df_scaled, c(13, 7, 3)) # 0.7912, 0.7802
set.seed(seed)
nn_model(train_df_scaled, test_df_scaled, c(13, 7)) # 0.8132, 0.8022
set.seed(seed)
nn_model(train_df_scaled, test_df_scaled, c(7, 3)) # 0.7692, 0.7582
set.seed(seed)
nn_model(train_df_scaled, test_df_scaled, c(7)) # 0.7912, 0.8043
# Neural Network with partially scaled data
set.seed(seed)
nn_model(train_df_p_scaled, test_df_p_scaled, c(13, 7, 3)) # 0.7692, 0.7582
set.seed(seed)
nn_model(train_df_p_scaled, test_df_p_scaled, c(13, 7)) # 0.7473, 0.7363
set.seed(seed)
nn_model(train_df_p_scaled, test_df_p_scaled, c(7, 3)) # 0.7253, 0.7582
set.seed(seed)
nn_model(train_df_p_scaled, test_df_p_scaled, c(7)) # 0.7473, 0.7473