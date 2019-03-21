#Read train set data
train_set <- read.csv("/Users/varshilaredkar/Downloads/BlogFeedback/blogData_train.csv", 
	header = FALSE)
#Directory path for test data files
file_paths <- list.files(path = "/Users/varshilaredkar/Downloads/BlogFeedback", pattern = "^blogData_test*.")

#Model for basic features
basic_features_train <- train_set[, 51:60]
basic_feature_model <- lm(train_set[, 281] ~ ., data = basic_features_train)
#Summary of Basic feature model
summary(basic_feature_model)

#Predict and calculate MSE for test files
mean_vector <- vector()
for (k in 1:length(file_paths)) {
	complete_path <- paste("/Users/varshilaredkar/Downloads/BlogFeedback/", file_paths[k])
	complete_path <- gsub(" ", "", complete_path)
	test_set <- read.csv(complete_path, header = FALSE)
	basic_features_test <- data.frame(test_set[, 51:60])
	predictions <- predict(basic_feature_model, newdata = basic_features_test)
	mse <- mean((test_set[, 281] - predictions)^2)
	mean_vector <- c(mean_vector, mse)
}

#Print MSEs for all files
print(mean_vector)
#Mean of all MSEs
print("MSE for basic features model")
print(mean(mean_vector))

#Model for textual features
textual_features_train <- train_set[, 63:262]
textual_feature_model <- lm(train_set[, 281] ~ ., data = textual_features_train)
#Summary of Textual feature model
summary(textual_feature_model)

#Predict and calculate MSE for test files
mean_vector1 <- vector()
for (k in 1:length(file_paths)) {
	complete_path <- paste("/Users/varshilaredkar/Downloads/BlogFeedback/", file_paths[k])
	complete_path <- gsub(" ", "", complete_path)
	test_set1 <- read.csv(complete_path, header = FALSE)
	textual_features_test <- data.frame(test_set1[, 63:262])
	predictions1 <- predict(textual_feature_model, newdata = textual_features_test)
	mse1 <- mean((test_set1[, 281] - predictions1)^2)
	mean_vector1 <- c(mean_vector1, mse1)
}

#Print MSEs for all files
print(mean_vector1)
#Mean of all MSEs
print("MSE for textual features model")
print(mean(mean_vector1))
