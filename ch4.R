# Chapter 4

rm(list = ls()); gc()
library(keras)
library(tidyverse)

train_data <- dataset_imdb(num_words = 10000)$train$x
train_labels <- dataset_imdb()$train$y
test_data <- dataset_imdb(num_words = 10000)$test$x
test_labels <- dataset_imdb()$test$y

vectorize_tokens <- function(tokens, dim2 = 10000){
  output_matrix <- matrix(0, nrow = length(tokens), ncol = dim2)
  for(i in 1:length(tokens))
    output_matrix[i, tokens[[i]]] <- 1
  output_matrix
}

x_train <- vectorize_tokens(train_data)
x_test <- vectorize_tokens(test_data)

val_split <- sample(1:25000, size = 0.2 * 25000)
x_val <- x_train[val_split, ]
y_val <- train_labels[val_split]
partial_x_train <- x_train[-val_split, ]
partial_y_train <- train_labels[-val_split]

# Original model
model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = 10000) %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid") %>% 
  compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = "accuracy"
  )

history <- model %>% 
  fit(
    partial_x_train,
    partial_y_train,
    epochs = 20,
    batch_size = 128,
    validation_data = list(x_val, y_val)
  )

# Model with lower capacity
model <- keras_model_sequential() %>% 
  layer_dense(units = 4, activation = "relu", input_shape = 10000) %>% 
  layer_dense(units = 4, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid") %>% 
  compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = "accuracy"
  )

history_lower <- model %>% 
  fit(
    partial_x_train,
    partial_y_train,
    epochs = 20,
    batch_size = 128,
    validation_data = list(x_val, y_val)
  )

# Model with higher capacity
model <- keras_model_sequential() %>% 
  layer_dense(units = 512, activation = "relu", input_shape = 10000) %>% 
  layer_dense(units = 512, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid") %>% 
  compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = "accuracy"
  )

history_higher <- model %>% 
  fit(
    partial_x_train,
    partial_y_train,
    epochs = 20,
    batch_size = 128,
    validation_data = list(x_val, y_val)
  )

tibble(network = rep(c("original", "lower", "higher"), each = 20),
       epoch = rep(1:20, times = 3),
       val_loss = c(history$metrics$val_loss, 
                    history_lower$metrics$val_loss, 
                    history_higher$metrics$val_loss)
       ) %>% 
  ggplot(aes(x = epoch, y = val_loss, color = network)) +
  geom_point() +
  theme_classic()

# L2 regularization

model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = 10000, kernel_regularizer = regularizer_l2(0.001)) %>% 
  layer_dense(units = 16, activation = "relu", kernel_regularizer = regularizer_l2(0.001)) %>% 
  layer_dense(units = 1, activation = "sigmoid") %>% 
  compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = "accuracy"
  )

history_l2 <- model %>% 
  fit(
    partial_x_train,
    partial_y_train,
    epochs = 20,
    batch_size = 128,
    validation_data = list(x_val, y_val)
  )

# Dropout regularization

model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = 10000) %>% 
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid") %>% 
  compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = "accuracy"
  )

history_dropout <- model %>% 
  fit(
    partial_x_train,
    partial_y_train,
    epochs = 20,
    batch_size = 128,
    validation_data = list(x_val, y_val)
  )
