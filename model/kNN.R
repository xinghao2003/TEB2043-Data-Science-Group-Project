library(mice)
library(dplyr)
library(dlookr)
library(caret)
library(class)

# Set seed for reproducibility
seed <- 123

# Load dataset from 'dataset.RData'
load("dataset.RData")

# Data Cleaning (if needed)
df <- unique(df)
md.pattern(df, rotate.names = TRUE)

# Create train/test sets (70/30 split) with set.seed
set.seed(seed)
r_train <- createDataPartition(df$output, p = 0.7, list = FALSE)
train_df <- df[r_train,]
test_df <- df[-r_train,]

# Standardize numeric features
numeric_features <- c("age", "trestbps", "chol", "thalach", "oldpeak")
train_df_std <- train_df
test_df_std <- test_df
set.seed(seed)  # Set seed for reproducibility
train_df_std[, numeric_features] <- scale(train_df_std[, numeric_features])
test_df_std[, numeric_features] <- scale(test_df_std[, numeric_features])

# KNN Model
k <- 5  # Choosing the number of neighbors
num_iterations <- 5  # Number of times to train the model

# Function to train and evaluate kNN model
train_evaluate_knn <- function(train_data, test_data, k_value) {
  set.seed(seed)  # Set seed for reproducibility
  knn_model <- knn(train = train_data[, -which(names(train_data) %in% "output")], 
                   test = test_data[, -which(names(test_data) %in% "output")], 
                   cl = train_data$output, k = k_value)
  accuracy <- mean(knn_model == test_data$output)
  return(accuracy)
}

# Original Features
accuracy_original <- numeric(num_iterations)
for (i in 1:num_iterations) {
  accuracy_original[i] <- train_evaluate_knn(train_df, test_df, k)
}
cat("Average Accuracy (Original Features):", mean(accuracy_original), "\n")  # Print average accuracy

# Standardized Features
accuracy_standardized <- numeric(num_iterations)
for (i in 1:num_iterations) {
  accuracy_standardized[i] <- train_evaluate_knn(train_df_std, test_df_std, k)
}
cat("Average Accuracy (Standardized Features):", mean(accuracy_standardized), "\n")  # Print average accuracy
