# Load necessary libraries
library(mice)      # For imputation
library(tidyverse) # Data manipulation
library(dplyr)     # Data manipulation
library(dlookr)    # Data analysis
library(caret)     # Machine learning toolkit
library(ggplot2)   # For plotting

library(randomForest) # Random Forest algorithm
library(doParallel)   # Parallel processing

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

# Data Preprocessing for Random Forest
df_p <- df
# Standardize numeric features
numeric_features <- c("age", "trtbps", "chol", "thalachh", "oldpeak")
df_p[numeric_features] <- scale(df_p[numeric_features])
# Convert categorical variables to factors
categorical_variables <- c("sex", "cp", "fbs", "restecg", "exng", "slp", "caa", "thall", "output")
df_p[categorical_variables] <- lapply(df_p[categorical_variables], factor)

# Create training and testing dataset
train_df_p <- df_p[r_train,]
test_df_p <- df_p[-r_train,]

# Train initial Random Forest model
set.seed(seed)
rf_initial <- randomForest(output ~ ., data = train_df_p)
accuracy_initial <- predict_accuracy(rf_initial, test_df_p, "Confusion Matrix - Initial Model (randomForest library)") # Check initial model accuracy

# Hyperparameter Tuning using caret, looking for the best parameters
# Testing parameter with random seed
set.seed(seed)
tuneLength <- 15
seeds <- lapply(1:16, function(round_num) sample.int(tuneLength^2, tuneLength))

# Set up control for repeated cross-validation
fit_control <- trainControl(
  method = "repeatedcv", 
  number = 5, 
  repeats = 3, 
  seeds = seeds, 
  search = "random",
  allowParallel = TRUE
)  # 5-fold cross-validation repeated 3 times, with random parameters

# Tuning the Random Forest model
set.seed(seed)
rf_tuned <- train(
  output ~ ., 
  data = train_df_p,
  method = "rf",
  trControl = fit_control,
  tuneLength = tuneLength
)

# Plot tuning results
trellis.par.set(caretTheme())
plot(rf_tuned, metric = "Accuracy", main = "Tuned RF (caret library)")

accuracy_tuned <- predict_accuracy(rf_tuned, test_df_p, "Confusion Matrix - Tuned RF (rf using caret library)")

# Model Retraining using best parameters from last step
# Extract best parameters
best_params <- rf_tuned$bestTune

# Retrain using best params
set.seed(seed)
rf_best <- randomForest(
  output ~ .,
  data = train_df_p,
  mtry = best_params$mtry
)
accuracy_best <- predict_accuracy(rf_best, test_df_p, "Confusion Matrix - Retrain using best params from Tuned RF (randomForest library)")

# Model Accuracy Comparison Plot
# Create a data frame with model names and their accuracies
model_names <- c("Initial Model (rf)", "Tuned RF (c)", "Best params Tuned RF (rf)")
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




# Get feature importance
importance_scores <- importance(rf_best)

# Extract MeanDecreaseGini scores and feature names
importance_values <- importance_scores[, "MeanDecreaseGini"]
feature_names <- rownames(importance_scores)

# Sort importance scores in increasing order
sorted_indices <- order(importance_values)
sorted_importance <- importance_values[sorted_indices]
sorted_feature_names <- feature_names[sorted_indices]

# Plotting the feature importance as a bar chart with inverted axes and vertical y-axis labels
barplot(sorted_importance, names.arg = sorted_feature_names, 
        main = "Final Model (Random Forest) Feature Importance",
        xlab = "Importance Score", ylab = "Features",
        col = "skyblue", horiz = TRUE, las = 2)

