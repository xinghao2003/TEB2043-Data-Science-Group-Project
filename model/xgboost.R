# Load necessary libraries
library(mice)      # For imputation
library(tidyverse) # Data manipulation
library(dplyr)     # Data manipulation
library(dlookr)    # Data analysis
library(caret)     # Machine learning toolkit
library(ggplot2)   # For plotting

library(xgboost)     # For xgboost, a type of ensemble model
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
predict_accuracy <- function(model, test_df, plot_title = "Confusion Matrix", matrix = FALSE) {
  if(matrix){
    X_test = test_df[ , !(names(test_df) %in% c('output'))]
    dtest <- xgb.DMatrix(data = as.matrix(X_test), label= test_df$output)
    
    predictions <- predict(model, dtest)
    predictions <- ifelse(predictions > 0.50, 1, 0)
    pred.matrix <- table(test_df$output, predictions)
  }else{
    predictions <- predict(model, test_df)
    pred.matrix <- table(test_df$output, predictions)
  }
  
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

# Data Preprocessing for xgboost Model
df_p <- df
# Create training and testing dataset
train_df_p <- df_p[r_train,]
test_df_p <- df_p[-r_train,]

# Preparing dataset for xgboost library
X_train = train_df_p[ , !(names(train_df_p) %in% c('output'))]                                             
y_train = train_df_p$output
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label= y_train)

negative_cases <- sum(y_train == 0)
positive_cases <- sum(y_train == 1)

# Training initial xgboost model
set.seed(seed)
xgb_initial <- xgboost(data = dtrain,       
                     max.depth = 4,
                     nround = 90,
                     objective = "binary:logistic",
                     scale_pos_weight = negative_cases/positive_cases,
                     gamma = 1
)
accuracy_initial <- predict_accuracy(xgb_initial, test_df_p, "Confusion Matrix - Initial Model", TRUE)


# Hyperparameter Tuning using caret, looking for the best parameters
# Testing parameter with random seed
set.seed(seed)
tuneLength <- 200
seeds <- lapply(1:16, function(round_num) sample.int(tuneLength^2, tuneLength))

# Preparing dataset for caret library
train_df_p$output <- factor(train_df_p$output, levels = c("0", "1"))

# Set up trainControl for cross-validation
fit_control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  seeds = seeds,
  search = "random",  
  allowParallel = TRUE
) # 5-fold cross-validation repeated 3 times, with random parameters

# Tuning the xgboost model using caret
xgb_tuned <- train(
  output ~ .,
  data = train_df_p,
  method = "xgbTree", 
  trControl = fit_control,
  tuneLength = tuneLength,
  metric = "Accuracy"
)
accuracy_tuned <- predict_accuracy(xgb_tuned, test_df_p, "Confusion Matrix - Tuned xgboost (caret library)")

# Plot top configurations in a scatter plot 
top_configs <- xgb_tuned$results %>%
  arrange(desc(Accuracy)) %>%
  head(20) 

ggplot(top_configs, aes(x = max_depth, y = nrounds, color = Accuracy)) +
  geom_point() +
  scale_color_gradient(low = "blue", high = "red") +
  labs(title = "Tuned xgboost Top 20 Configurations:", x = "C", y = "sigma", color = "Accuracy") +
  theme_minimal()

# Model Retraining using best parameters from last step
# Extract best parameters
best_params <- xgb_tuned$bestTune

# Prepare for xgboost library
train_df_p <- df_p[r_train,]

set.seed(seed)
xgb_best <- xgboost(
  data = dtrain,
  objective = "binary:logistic",
  max.depth = best_params$max_depth,  
  nround = best_params$nrounds,  
  gamma = best_params$gamma,  
  eta = best_params$eta,  
  colsample_bytree = best_params$colsample_bytree,  
  min_child_weight = best_params$min_child_weight, 
  subsample = best_params$subsample, 
  scale_pos_weight = negative_cases / positive_cases,
)
accuracy_best <- predict_accuracy(xgb_best, test_df_p, "Confusion Matrix - Retrain using best params from Tuned xgboost", TRUE)

# Model Accuracy Comparison Plot
# Create a data frame with model names and their accuracies
model_names <- c("Initial Model (xgboost)", "Tuned xgboost (c)", "Best params (xgboost)")
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

