library(mice) # library for impute
library(tidyverse)
library(dplyr)
library(dlookr)
library(caret) 

library(rpart) # For decision tree model
library(rpart.plot) # For data visualization

# Load variables into global environments from dataset.RData
# These variables were prepared in dataset.R
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
  # Fit A Classification Tree
  fit.tree = rpart(output ~ ., data=train, method = "class", control = control)
  # Visualizing the unpruned tree
  rpart.plot(fit.tree)
  # Checking the order of variable importance
  print(fit.tree$variable.importance)
  # Predict Using the unpruned tree
  pred.tree = predict(fit.tree, test, type = "class")
  # Evaluate the Performance of the Classification Tree
  pred.tree.matrix = table(pred.tree, test$output)
  print(confusionMatrix(pred.tree.matrix))
  plotcp(fit.tree)
  printcp(fit.tree)
  # Explicitly request the lowest cp value
  print(fit.tree$cptable[which.min(fit.tree$cptable[,"xerror"]),"CP"])
  # Selecting the lowest cp value, and fit a classification tree
  bestcp <- fit.tree$cptable[which.min(fit.tree$cptable[,"xerror"]),"CP"]
  pruned.tree <- prune(fit.tree, cp = bestcp)
  # Visualizing the pruned tree
  rpart.plot(pruned.tree)
  # Predict Using the pruned tree
  pred.prune = predict(pruned.tree, test, type="class")
  # Evaluate the Performance of the Classification Tree
  pred.prune.matrix = table(pred.prune, test$output)
  print(confusionMatrix(pred.tree.matrix))
}

# Setting seed for reproducible
set.seed(seed)

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
sapply(df_p,class)
train_df_p = df_p[r_train,]
test_df_p = df_p[-r_train,]

# Classification Decision Tree
control <- rpart.control(cp = 0.0001)
class_tree(train_df, test_df, control) # 0.8
class_tree(train_df_p, test_df_p, control) # 0.8
