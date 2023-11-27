library(mice)
library(tidyverse)
library(dplyr)
library(dlookr)
library(caret)
library(rpart)
library(rpart.plot)

# Load variables into global environments from dataset.RData
load("dataset/dataset.RData")

# Predict from model and calculate its accuracy
predict_accuracy <- function(model, test_df) {
  predictions <- predict(model, test_df)
  plot(predictions)
  pred.matrix = table(test_df$output, predictions)
  print(confusionMatrix(pred.matrix))
}

# Classification Decision Tree Model
class_tree <- function(train, test, control = rpart.control(cp = 0.008)) {
  fit.tree = rpart(output ~ ., data = train, method = "class", control = control)
  rpart.plot(fit.tree)
  print(fit.tree$variable.importance)
  pred.tree = predict(fit.tree, test, type = "class")
  pred.tree.matrix = table(pred.tree, test$output)
  print(confusionMatrix(pred.tree.matrix))
  plotcp(fit.tree)
  printcp(fit.tree)
  print(fit.tree$cptable[which.min(fit.tree$cptable[,"xerror"]), "CP"])
  bestcp <- fit.tree$cptable[which.min(fit.tree$cptable[,"xerror"]), "CP"]
  pruned.tree <- prune(fit.tree, cp = bestcp)
  rpart.plot(pruned.tree)
  pred.prune = predict(pruned.tree, test, type = "class")
  pred.prune.matrix = table(pred.prune, test$output)
  print(confusionMatrix(pred.tree.matrix))
  return(list(model = pruned.tree, accuracy = confusionMatrix(pred.prune.matrix)$overall["Accuracy"]))
}

# Setting seed for reproducible
set.seed(seed)

# Number of iterations
num_iterations <- 5
models <- list()

for (i in 1:num_iterations) {
  # Generate partition for train and test dataset
  r_train <- createDataPartition(df$output, p = 0.7, list = FALSE)
  
  train_df = df[r_train,]
  test_df = df[-r_train,]
  
  # Standardize numeric features
  df_p <- df
  numeric_features <- c("age", "trtbps", "chol", "thalachh", "oldpeak")
  df_p[numeric_features] <- scale(df_p[numeric_features])
  # Convert categorical variables to factors
  categorical_variables <- c("sex", "cp", "fbs", "restecg", "exng", "slp", "caa", "thall", "output")
  df_p[categorical_variables] <- lapply(df_p[categorical_variables], factor)
  sapply(df_p, class)
  train_df_p = df_p[r_train,]
  test_df_p = df_p[-r_train,]
  
  # Classification Decision Tree
  control <- rpart.control(cp = 0.0001)
  result <- class_tree(train_df_p, test_df_p, control)
  models[[i]] <- result
}

# Display the accuracy of each model
for (i in 1:num_iterations) {
  cat("Accuracy for Model", i, ":", models[[i]]$accuracy, "\n")
}

#Accuracy for Model 1 : 0.8 
#Accuracy for Model 2 : 0.7333333 
#Accuracy for Model 3 : 0.6777778 
#Accuracy for Model 4 : 0.7333333 
#Accuracy for Model 5 : 0.8