# Load necessary libraries
library(mice)      # For imputation
library(tidyverse) # Data manipulation
library(dplyr)     # Data manipulation
library(dlookr)    # Data analysis
library(caret)     # Machine learning toolkit
library(ggplot2)   # For plotting

library(kernlab)     # For Support Vector Machine
library(doParallel)  # Parallel processing

# Parallel processing setup
# Set the number of cores/workers for parallel processing
num_cores <- 8  # Set based on available cores

# Initialize parallel backend using doParallel
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Data loading and initialize the environments
# Load a pre-config global environment
load("dataset/dataset.RData")

# Function to calculate model accuracy
predict_accuracy <- function(model, test_df, plot_title) {
  predictions <- predict(model, test_df)
  pred.matrix <- table(test_df$output, predictions)
  
  cm <- confusionMatrix(pred.matrix)
  acc <- cm$overall['Accuracy']
  
  # Convert confusion matrix to a data frame for ggplot
  cm_df <- as.data.frame(pred.matrix)
  
  # Create the confusion matrix plot using ggplot2
  plot <- ggplot(data = cm_df, aes(x = predictions, y = Var1, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), vjust = 0.5, color = "white") +
    labs(title = plot_title, subtitle = paste("Accuracy:", round(acc, 4)), x = "Predicted", y = "Actual") +
    theme_minimal() +
    scale_fill_gradient(low = "lightblue", high = "darkblue")
  
  print(plot)
  
  print(cm)  # Print confusion matrix
  
  return(acc)
}

# Data Preprocessing for SVM Model
df_p <- df
# Standardize numeric features
numeric_features <- c("age", "trtbps", "chol", "thalachh", "oldpeak")
df_p[numeric_features] <- scale(df_p[numeric_features])
# Convert categorical variables to factors
categorical_variables <- c("sex", "cp", "fbs", "restecg", "exng", "slp", "caa", "thall")
df_p[categorical_variables] <- lapply(df_p[categorical_variables], factor)
# One-hot encoding for all factors except output
data_encoded <- dummyVars(" ~ .", data = df_p)
df_p <- data.frame(predict(data_encoded, newdata = df_p))
# Convert categorical variable output to factors
df_p$output <- factor(df_p$output, levels = c("0", "1"))

# Create training and testing dataset
train_df_p <- df_p[r_train,]
test_df_p <- df_p[-r_train,]

# Create initial SVM model
set.seed(seed)
svm_initial <- ksvm(
  output ~ ., 
  data = train_df_p, 
  type = "C-svc", 
  kernel = "rbfdot"
  )
accuracy_initial <- predict_accuracy(svm_initial, test_df_p, "Confusion Matrix - Initial Model") 

# Hyperparameter Tuning using caret, looking for the best parameters
# Testing parameter with random seed
set.seed(seed)
tuneLength <- 1000
seeds <- lapply(1:16, function(round_num) sample.int(tuneLength^2, tuneLength))

# Define cross-validation method
fit_control <- trainControl(
  method = "repeatedcv", 
  number = 5, 
  repeats = 3, 
  seeds = seeds,
  search = "random",
  allowParallel = TRUE
) # 5-fold cross-validation repeated 3 times, with random parameters

# Tuning the SVM model
svm_tuned <- train(
  output ~ ., 
  data = train_df_p, 
  method = "svmRadial",  # Use radial SVM
  trControl = fit_control,  # Use the defined train control
  tuneLength = tuneLength
)  

# Plot top configurations in a scatter plot (C vs. sigma)
top_configs <- svm_tuned$results %>%
  arrange(desc(Accuracy)) %>%
  head(100) 

ggplot(top_configs, aes(x = C, y = sigma, color = Accuracy)) +
  geom_point() +
  scale_color_gradient(low = "blue", high = "red") +
  labs(title = "Tuned SVM Top 100 Configurations: C vs. sigma", x = "C", y = "sigma", color = "Accuracy") +
  theme_minimal()

accuracy_tuned <- predict_accuracy(svm_tuned, test_df_p, "Confusion Matrix - Tuned SVM (svm using caret library)")

# Model Retraining using best parameters from last step
# Extract best parameters
best_params <- svm_tuned$bestTune

# Retrain using best params
set.seed(seed)
svm_best <- ksvm(
  output ~ ., 
  data = train_df_p, 
  type = "C-svc", 
  kernel = "rbfdot",
  C = best_params$C,
  kpar = list(sigma = best_params$sigma)
)
accuracy_best <- predict_accuracy(svm_best, test_df_p, "Confusion Matrix - Retrain using best params from Tuned SVM (svm library)") 

# Model Accuracy Comparison Plot
# Create a data frame with model names and their accuracies
model_names <- c("Initial Model (svm)", "Tuned SVM (c)", "Best params (svm)")
accuracies <- c(accuracy_initial, accuracy_tuned, accuracy_best)
accuracy_df <- data.frame(Model = model_names, Accuracy = accuracies)

# Create the bar plot comparing model accuracies with adjusted x-axis text and accuracy labels
accuracy_plot <- ggplot(accuracy_df, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(round(Accuracy, 3) * 100, "%")), 
            position = position_dodge(width = 0.9), 
            vjust = -0.5, size = 3) +  # Add accuracy labels above bars
  labs(title = "Comparison of Model Accuracies", x = "Model", y = "Accuracy") +
  theme_minimal() 

# Show the plot
print(accuracy_plot)

