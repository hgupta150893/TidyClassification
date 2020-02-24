library(dslabs)
library(tidyverse)
library(broom)
library(rsample)
library(ranger)
library(Metrics)

# Reading the Data

attrition <- read_csv("Data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Removing Unnecessary columns

attrition <- attrition %>% select(-EmployeeCount,-EmployeeNumber,-StandardHours,-Over18)

# Creating Target Column
attrition$Attrition <- attrition$Attrition == "Yes"
table(attrition$Attrition)

# Creating Train & Test  Split

set.seed(42)

split <- initial_split(attrition,prop = 0.75)
training_data <- training(split)
testing_data <- testing(split)

# Creating Cross Validation Split within Training Data

cv_split <- vfold_cv(training_data,v=5)

cv_data <- cv_split %>% 
    mutate(
        # Extract the train dataframe for each split
        train = map(splits, ~training(.x)), 
        # Extract the validate dataframe for each split
        validate = map(splits, ~testing(.x))
    )


# Creating Logistic Regression Model

cv_models_lr <- cv_data %>% 
    mutate(model = map(train, 
                       ~glm(formula = Attrition~., data = .x, family = "binomial")))

# Evaluating the model

    # Validating 1st Model

    model <- cv_models_lr$model[[1]]
    validate <- cv_models_lr$validate[[1]]
    
    # Actual Values
    
    validate_actual <- validate$Attrition
    
    # Predicted Values
    
    validate_prob <- predict(model, validate, type = "response")
    validate_predicted <- validate_prob > 0.5
    
    # Calculating Metrics
    
    table(validate_actual, validate_predicted)
    
    accuracy(validate_actual, validate_predicted)
    precision(validate_actual, validate_predicted)
    recall(validate_actual, validate_predicted)

# Scaling the Evaluation for All the Models
    
cv_prep_lr <- cv_models_lr %>% 
    mutate(
        # Extract the recorded life expectancy for the records in the validate dataframes
        validate_actual = map(validate, ~.x$Attrition),
        # Predict life expectancy for each validate set using its corresponding model
        validate_predicted = map2(.x = model, .y = validate, ~predict(.x, .y, type = "response") >0.5))


cv_perf_recall <- cv_prep_lr %>% 
    mutate(validate_recall = map2_dbl(.x = validate_actual, .y = validate_predicted, 
                                      ~recall(actual = .x, predicted = .y)))

# Print the validate_recall column
cv_perf_recall$validate_recall

# Calculate the average of the validate_recall column
mean(cv_perf_recall$validate_recall)

## Mean Recall is 0.40 for Logistic Regression model

##################################################################
# Model 2
##################################################################


# Creating Random Forest Model

# Setup Hyperparameters for testing
cv_tune <- cv_data %>%
    crossing(mtry = c(2,4,8,16)) 

cv_models_rf <- cv_tune %>% 
    mutate(model = map2(.x = train, .y = mtry, 
                       ~ranger(formula = Attrition ~ ., data = .x, mtry = .y, num.trees = 100, seed = 42)))

# Calculating Metrics

cv_prep_rf <- cv_models_rf %>% 
    mutate(
        # Prepare binary vector of actual Attrition values in validate
        validate_actual = map(validate, ~.x$Attrition),
        # Prepare binary vector of predicted Attrition values for validate
        validate_predicted = map2(.x = model, .y = validate, ~predict(.x, .y, type = "response")$predictions)
    )

# Calculate the validate recall for each cross validation fold
cv_perf_recall <- cv_prep_rf %>% 
    mutate(recall = map2_dbl(.x = validate_actual, .y = validate_predicted, ~recall(actual = .x, predicted = .y)))

# Calculate the mean recall for each mtry used  
cv_perf_recall %>% 
    group_by(mtry) %>% 
    summarise(mean_recall = mean(recall))


## Random Forest performance is not giving us good recall
## The best value is ~0.22
## We will build logistic regression model for the entire data

##################################################################
# Full Model
##################################################################

# Build the logistic regression model using all training data

best_model <- glm(formula = Attrition~., 
                  data = training_data, family = "binomial")


# Prepare binary vector of actual Attrition values for testing_data
test_actual <- testing_data$Attrition

# Prepare binary vector of predicted Attrition values for testing_data
test_predicted <- predict(best_model, testing_data, type = "response") > 0.5

table(test_actual,test_predicted)
accuracy(test_actual,test_predicted)
precision(test_actual,test_predicted)
recall(test_actual,test_predicted)

# Recall of 0.52 is achieved on the full dataset
# Best model object can be exported for future usage
