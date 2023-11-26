library(mice)
library(dplyr)
library(dlookr)
library(caret)

# Set seed for reproducibility
seed <- 123

# Read 'heart.csv' into 'df'
df <- read.csv("dataset/heart.csv")

# Summary and class of variables
summary(df)
sapply(df, class)

# Data Cleaning
df <- unique(df)  # Remove duplicates
md.pattern(df, rotate.names = TRUE)  # Check for missing values

# Data Exploration
# Univariate analysis
describe(df)
normality(df)  # Test normality
plot_normality(df)  # Visualize normality

# Bivariate/multivariate analysis
correlate(df)  # Correlation coefficient
df %>% correlate() %>% plot()  # Visualization of correlation matrix

# Create train/test sets (70/30 split)
set.seed(seed)
r_train <- createDataPartition(df$output, p = 0.7, list = FALSE)
train_df <- df[r_train,]
test_df <- df[-r_train,]

# Store data into 'dataset.RData'
save(df, seed, r_train, train_df, test_df, file = "dataset/dataset.RData")
