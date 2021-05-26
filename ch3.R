# Chapter 3
library(keras)
library(tidyverse)

# IMDB example

imdb <- dataset_imdb(num_words = 10000)

train_data <- imdb$train$x
test_data <- imdb$test$x
train_labels <- imdb$train$y
test_labels <- imdb$test$y

str(train_data[[1]])
str(train_labels[[1]])

word_index <- dataset_imdb_word_index()
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index
decode_review <- function(index){
  sapply(train_data[[index]], function(x){
    word <- if(x >= 3)
      word <- reverse_word_index[[as.character(x - 3)]]
    if(!is.null(word))
      word
    else
      "?"
  })
}
decode_review(2)

vectorize_sequences <- function(sequences, dimension = 10000){
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for(i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1
  results
}

x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = 10000) %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>%
  compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = "accuracy"
  )

val_indices <- 1:10000
x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]
y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]

history <- model %>% 
  fit(
    partial_x_train,
    partial_y_train,
    epochs = 4,
    batch_size = 512,
    validation_data = list(x_val, y_val)
  )

results <- model %>% evaluate(x_test, y_test)

# Newswire example

rm(list = ls())
gc()

reuters <- dataset_reuters(num_words = 10000)
str(reuters)
train_data <- reuters$train$x
train_labels <- reuters$train$y
test_data <- reuters$test$x
test_labels <- reuters$test$y
word_index <- dataset_reuters_word_index()

vectorize_sequences <- function(sequences, dimension = 10000){
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for(i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1
  results
}

x_train <- vectorize_sequences(train_data)
y_train <- vectorize_sequences(test_data)


to_one_hot <- function(labels, dimension = 46){
  results <- matrix(0, nrow = length(labels), ncol = dimension)
  for(i in 1:length(labels))
    results[i, labels[[i]]] <- 1
  results
}

one_hot_train_labels <- to_one_hot(train_labels) # this is equivalent to keras::to_categorical(train_labels)
one_hot_test_labels <- to_one_hot(test_labels)

model <- keras_model_sequential() %>% 
  layer_dense(units = 64, activation = "relu", input_shape = 10000) %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 46, activation = "softmax")

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)

val_indices <- 1:1000
x_val <- x_train[val_indices, ]
y_val <- one_hot_train_labels[val_indices, ]
partial_x_train <- x_train[-val_indices, ]
partial_y_train <- one_hot_train_labels[-val_indices, ]

history <- model %>% 
  fit(
    partial_x_train, 
    partial_y_train, 
    epochs = 20,
    batch_size = 512,
    validation_data = list(x_val, y_val)
  )

model <- keras_model_sequential() %>% 
  layer_dense(units = 64, activation = "relu", input_shape = 10000) %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 46, activation = "softmax")

model %>% 
  compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = "accuracy"
  )

history <- model %>% 
  fit(
    partial_x_train,
    partial_y_train,
    epochs = 9,
    batch_size = 512,
    validation_data = list(x_val, y_val)
  )

x_test <- vectorize_sequences(test_data)

results <- model %>% 
  evaluate(x_test,
           one_hot_test_labels)

predictions <- model %>% 
  predict(x_test)
dim(predictions)
sum(predictions[3,])
which.max(predictions[1,])

model <- keras_model_sequential() %>% 
  layer_dense(units = 32, activation = "relu", input_shape = 10000) %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 46, activation = "softmax")

model %>% 
  compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = "accuracy"
  )

history <- model %>% 
  fit(
    partial_x_train,
    partial_y_train,
    epochs = 6,
    batch_size = 128,
    validation_data = list(x_val, y_val)
  )

model %>% evaluate(x_test, one_hot_test_labels)

# Regression problem example

rm(list = ls())

housing <- dataset_boston_housing()
train_data <- housing$train$x
test_data <- housing$test$x
train_targets <- housing$train$y
test_targets <- housing$test$y

str(train_data)
str(train_targets)

means <- apply(train_data, 2, mean)
sds <- apply(train_data, 2, sd)
train_data <- apply(train_data, 2, scale)
test_data <- scale(test_data, center = means, scale = sds)

build_model <- function(){
  model <- keras_model_sequential() %>% 
    layer_dense(units = 64, activation = "relu", input_shape = dim(train_data)[2]) %>% 
    layer_dense(units = 64, activation = "relu") %>% 
    layer_dense(units = 1)
  
  model %>%
    compile(
      optimizer = "rmsprop",
      loss = "mse",
      metrics = "mae"
    )
}

k <- 4
indices <- sample(1:nrow(train_data))
folds <- cut(1:length(indices), breaks = k, labels = F) # make factor labels 1-4 for all indices 1:404
num_epochs <- 500
all_mae_histories <- NULL
for(i in 1:k){
  cat("Processing fold #", i, "\n")
  
  val_indices <- which(folds == i, arr.ind = T)
  val_data <- train_data[val_indices, ]
  val_targets <- train_targets[val_indices]
  
  partial_train_data <- train_data[-val_indices, ]
  partial_train_targets <- train_targets[-val_indices]
  
  model <- build_model()
  
  history <- model %>% 
    fit(
      partial_train_data,
      partial_train_targets,
      epochs = num_epochs,
      batch_size = 1,
      verbose = 0,
      validation_data = list(val_data, val_targets)
    )
  
  mae_history <- history$metrics$val_mae
  all_mae_histories <- rbind(all_mae_histories, mae_history)
    
}

all_mae_histories %>% 
  t %>%
  as_tibble() %>%
  mutate(epoch = row_number()) %>%
  rename(fold_1 = 1, fold_2 = 2, fold_3 = 3, fold_4 = 4) %>% 
  rowwise() %>%
  mutate(average_mae = mean(c(fold_1, fold_2, fold_3, fold_4))) %>%
  ungroup() %>%
  pivot_longer(names_to = "fold", values_to = "mae", -epoch) %>%
  ggplot(aes(x = epoch, y = mae, color = fold)) +
  geom_smooth() +
  theme_classic()









