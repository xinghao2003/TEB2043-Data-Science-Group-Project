library(mice) # library for impute
library(tidyverse)
library(dplyr)
library(dlookr)
library(caret) 

#install.packages("C50")
library(C50)

load("dataset/dataset.RData")

# Predict from model and calculate its accuracy
predict_accuracy <- function(model, test_df) {
  predictions <- predict(model, test_df)
  plot(predictions)
  pred.matrix = table(test_df$output, predictions)
  print(confusionMatrix(pred.matrix))
}

set.seed(seed)

# Pre-process the data set for rule induction
df_p <- df
# Standardize numeric features
numeric_features <- c("age", "trtbps", "chol", "thalachh", "oldpeak")
df_p[numeric_features] <- scale(df_p[numeric_features])
# Convert categorical variables to factors
categorical_variables <- c("sex", "cp", "fbs", "restecg", "exng", "slp", "caa", "thall", "output")
df_p[categorical_variables] <- lapply(df_p[categorical_variables], factor)

sapply(df_p,class)

train_df_p = df_p[r_train,]
test_df_p = df_p[-r_train,]

# Train the model using C5.0
model <- C5.0(output ~ ., data = train_df_p)

# Summary of the model
summary(model)

# Predict on test data
predict_accuracy(model, test_df_p)
