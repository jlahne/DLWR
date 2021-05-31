# Chapter 6

library(keras)
library(tidyverse)

# Skipping the one-hot coding examples because they are quite tedious.  Startign with how to use built in keras::text_tokenizer()

samples <- c("The cat sat on the mat", "The dog ate my homework")

tokenizer <- text_tokenizer(num_words = 1000) %>% 
  fit_text_tokenizer(samples)

sequences <- texts_to_sequences(tokenizer, samples) # This step produces the kind of format in the IMDB example from dataset_imdb()

one_hot_results <- texts_to_matrix(tokenizer, samples, mode = "binary")

word_index <- tokenizer$word_index

install.packages("hashFunction")

# IMBD embedding layer example

max_features <- 10000
maxlen <- 20

imdb <- dataset_imdb(num_words = max_features)
x_train <- imdb$train$x
y_train <- imdb$train$y
x_test <- imdb$test$x
y_test <- imdb$test$y

x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)

model_embedding <- keras_model_sequential() %>% 
  layer_embedding(input_dim = 10000, output_dim = 8, input_length = maxlen) %>%
  layer_flatten() %>%
  layer_dense(units = 1, activation = "sigmoid") %>% 
  compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = "acc"
  )

summary(model_embedding)

history <- model_embedding %>% 
  fit(
    x_train, 
    y_train,
    epochs = 10,
    batch_size = 32,
    validation_split = 0.2
  )

evaluate(model_embedding, x_test, y_test)

# Using pretrained embeddings - GloVe

rm(list = ls()); gc()

imdb_dir <- "~/Downloads/aclImdb"
train_dir <- file.path(imdb_dir, "train")

labels <-c()
texts <- c()

for(label_type in c("neg", "pos")){
  label <- switch(label_type, neg = 0, pos = 1)
  dir_name <- file.path(train_dir, label_type)
  for(fname in list.files(dir_name, pattern = glob2rx("*.txt"), full.names = TRUE)){
    texts <- c(texts, readChar(fname, file.info(fname)$size))
    labels <- c(labels, label)
  }
}

maxlen <- 100  
training_samples <- 200
validation_samples <- 10000
max_words <- 10000

tokenizer <- text_tokenizer(num_words = max_words) %>% 
  fit_text_tokenizer(texts)

sequences <- texts_to_sequences(tokenizer, texts)
sequences[[1]]

word_index <- tokenizer$word_index
length(word_index)

data <- pad_sequences(sequences, maxlen = maxlen)
labels <- as.array(labels)  
dim(data)  
dim(labels)  

indices <- sample(1:nrow(data))  
training_indices <- indices[1:training_samples]
validation_indices <- indices[(training_samples + 1):(validation_samples + training_samples)]

x_train <- data[training_indices, ]
y_train <- labels[training_indices]
x_val <- data[validation_indices, ]
y_val <- labels[validation_indices]

glove_dir <- "~/Documents/R/Deep Learning with R/glove.6B"
lines <- readLines(file.path(glove_dir, "glove.6B.100d.txt"))
lines[1]
lines[2]  

embeddings_index <- new.env(hash = TRUE, parent = emptyenv())  

for(i in 1:length(lines)){
  line <- lines[i]
  values <- strsplit(line, " ")[[1]]
  word <- values[1]
  embeddings_index[[word]] <- as.double(values[-1])
}

embedding_dim <- 100

embedding_matrix <- array(0, dim = c(max_words, embedding_dim))

for(word in names(word_index)){
  index <- word_index[[word]]
  if(index < max_words){
    embedding_vector <- embeddings_index[[word]]
    if(!is.null(embedding_vector))
      embedding_matrix[index + 1, ] <- embedding_vector
  }
}
embedding_matrix[1,]
word_index[1]
embedding_matrix[2,]
embeddings_index[["the"]]

model_glove <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_words, output_dim = embedding_dim, input_length = maxlen) %>% 
  layer_flatten() %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model_glove  

get_layer(model_glove, index = 1) %>%
  set_weights(list(embedding_matrix)) %>% 
  freeze_weights

model_glove  

model_glove %>% 
  compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = "acc"
  )

history <- model_glove %>% 
  fit(
    x_train,
    y_train,
    epochs = 20,
    batch_size = 32,
    validation_data = list(x_val, y_val)
  )
plot(history)  
save_model_hdf5(model_glove, "pre_trained_glove_model.h5")  

model_noglove <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_words, output_dim = embedding_dim, input_length = maxlen) %>% 
  layer_flatten() %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid") %>% 
  compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = "acc"
  )

history <- model_noglove %>% 
  fit(
    x_train,
    y_train,
    epochs = 20,
    batch_size = 32,
    validation_data = list(x_val, y_val)
  )

test_dir <- file.path(imdb_dir, "test")

labels <- c()
texts <- c()

for(label_type in c("neg", "pos")){
  label <- switch(label_type, neg = 0, pos = 1)
  dir_name <- file.path(test_dir, label_type)
  for(fname in list.files(dir_name, pattern = glob2rx("*.txt"), full.names = TRUE)){
    texts <- c(texts, readChar(fname, file.info(fname)$size))
    labels <- c(labels, label)
  }
}

sequences <- texts_to_sequences(tokenizer, texts)
x_test <- pad_sequences(sequences, maxlen = maxlen)
y_test <- as.array(labels)  

model_glove %>% evaluate(x_test, y_test)
model_noglove %>% evaluate(x_test, y_test)  
  
# RNNs - reset R here

rm(list = ls()); gc()
library(keras)
library(tidyverse)

timesteps <- 100
input_features <- 32
output_features <- 64

random_array <- function(dim){
  array(runif(prod(dim)), dim = dim)
}

inputs <- random_array(dim = c(timesteps, input_features))
state_t <- rep_len(0, length = output_features)

W <- random_array(dim = c(output_features, input_features))
U <- random_array(dim = c(output_features, output_features))
b <- random_array(dim = c(output_features, 1))

output_sequence <- array(0, dim = c(timesteps, output_features))
for(i in 1:nrow(inputs)){
  input_t <- inputs[i, ]
  output_t <- tanh(
    as.numeric(
      (W %*% input_t) + (U %*% state_t) + b
    )
  )
  output_sequence[i,] <- as.numeric(output_t)
  state_t <- output_t
}
output_sequence

rm(list = ls()); gc()

max_features <- 10000
maxlen = 500
batch_size = 32

imdb <- dataset_imdb(num_words = max_features)
input_train <- imdb$train$x
y_train <- imdb$train$y
input_test <- imdb$test$x
y_test <- imdb$test$y

input_train <- pad_sequences(input_train, maxlen = maxlen)
input_test <- pad_sequences(input_test, maxlen = maxlen)
dim(input_train)
dim(input_test)

model_rnn <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_features, output_dim = 32) %>% 
  layer_simple_rnn(units = 32) %>% 
  layer_dense(units = 1, activation = "sigmoid") %>% 
  compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = "acc"
  )
model_rnn

history <- model_rnn %>% 
  fit(
    input_train,
    y_train,
    epochs = 10,
    batch_size = 128,
    validation_split = 0.2
  )
evaluate(model_rnn, input_test, y_test)

model_lstm <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_features, output_dim = 32) %>% 
  layer_lstm(units = 32) %>% 
  layer_dense(units = 1, activation = "sigmoid") %>% 
  compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = "acc"
  )

history <- model_lstm %>% 
  fit(
    input_train,
    y_train,
    epochs = 10,
    batch_size = 128,
    validation_split = 0.2
  )
  
evaluate(model_lstm, input_test, y_test)

# Advanced RNNs (and generators)--reset R here

library(keras)
library(tidyverse)

data_dir <- "~/Documents/R/Deep Learning with R/jena_climate"  
fname <- file.path(data_dir, "jena_climate_2009_2016.csv")  
data <- read_csv(fname)  

glimpse(data)

ggplot(data, aes(x = 1:nrow(data), y = `T (degC)`)) + geom_line()
ggplot(data[1:1440, ], aes(x = 1:1440, y = `T (degC)`)) + geom_line()

sequence_generator <- function(start){
  value <- start - 1
  function(){
    value <<- value + 1
    value
  }
}

gen <- sequence_generator(10)
gen()
gen()
gen()
gen2 <- sequence_generator(10)
gen()
gen2()
gen2

data <- data.matrix(data[, -1])
train_data <- data[1:200000, ]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(data, center = mean, scale = std)

# generator <- function(data, lookback, delay, min_index, max_index, shuffle = FALSE, batch_size = 128, step = 6){
#   
#   if(is.null(max_index))
#     max_index <- nrow(data) - delay - 1
#   i <- min_index + lookback
#   
#   function(){
#     if(shuffle){
#       rows <- sample(c((min_index + lookback):max_index), size = batch_size)
#     } else{
#       if(i + batch_size >= max_index)
#         i <<- min_index + lookback
#       rows <- c(i:min(i + batch_size - 1, max_index))
#       i <<- i + length(rows)
#     }
#     
#     samples <- array(0, dim = c(length(rows), lookback / step, dim(data)[[-1]]))
#     targets <- array(0, dim = length(rows))
#     
#     for(j in 1:length(rows)){
#       indices <- seq(rows[[j]] - lookback, rows[[j]] - 1, length.out = dim(samples)[[2]])
#       samples[j, , ] <- data[indices, ]
#       targets[[j]] <- data[rows[[j]] + delay, 2]
#     }
#     
#     list(samples, targets)
#     
#   }
#   
# }

generator <- function(data, lookback, delay, min_index, max_index,
                      shuffle = FALSE, batch_size = 128, step = 6) {
  if (is.null(max_index))
    max_index <- nrow(data) - delay - 1
  i <- min_index + lookback
  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (i + batch_size >= max_index)
        i <<- min_index + lookback
      rows <- c(i:min(i+batch_size-1, max_index))
      i <<- i + length(rows)
    }
    
    samples <- array(0, dim = c(length(rows),
                                lookback / step,
                                dim(data)[[-1]]))
    targets <- array(0, dim = c(length(rows)))
    
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]]-1,
                     length.out = dim(samples)[[2]])
      samples[j,,] <- data[indices,]
      targets[[j]] <- data[rows[[j]] + delay,2]
    }           
    list(samples, targets)
  }
}

lookback <- 1440
step <- 6
delay <- 144
batch_size <- 128

train_gen <- generator(data, delay = delay, lookback = lookback, min_index = 1, max_index = 200000, shuffle = TRUE, step = step, batch_size = batch_size)
val_gen <- generator(data, lookback = lookback, delay = delay, min_index = 200001, max_index = 300000, step = step, batch_size = batch_size, shuffle = FALSE)
test_gen <- generator(data, lookback = lookback, delay = delay, min_index = 300001, max_index = NULL, step = step, batch_size = batch_size, shuffle = FALSE)

val_steps <- (300000 - 200001 - lookback) / batch_size
test_steps <- (nrow(data) - 300001 - lookback) / batch_size

evaluate_naive_method <- function(){
  batch_maes <- c()
  for(step in 1:val_steps){
    c(samples, targets) %<-% val_gen()
    preds <- samples[, dim(samples)[[2]], 2]
    mae <- mean(abs(preds - targets))
    batch_maes <- c(batch_maes, mae)
  }
  print(mean(batch_maes))
}

evaluate_naive_method()

model_temp_simple <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(lookback / step, dim(data)[-1])) %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 1) %>% 
  compile(
    optimizer = "rmsprop",
    loss = "mae"
  )

history <- model_temp_simple %>% 
  fit_generator(
    train_gen,
    steps_per_epoch = 500,
    epochs = 20,
    validation_data = val_gen,
    validation_steps = val_steps
  )

model_temp_gru <- keras_model_sequential() %>%
  layer_gru(units = 32, input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_dense(units = 1) %>% 
  compile(
    optimizer = optimizer_rmsprop(),
    loss = loss_mean_absolute_error
  )

history <- model_temp_gru %>% 
  fit(
    train_gen,
    steps_per_epoch = 500,
    epochs = 20,
    validation_data = val_gen,
    validation_steps = val_steps
  )

# Bidirectional RNNs -- restart R here

rm(list = ls()); gc()
library(keras)
library(tidyverse)

max_features <- 10000
maxlen <- 500

c(c(x_train, y_train), c(x_test, y_test)) %<-% dataset_imdb(num_words = max_features)
x_train <- pad_sequences(x_train, maxlen)
x_test <- pad_sequences(x_test, maxlen)

model_imdb_bidirectional <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_features, output_dim = 32) %>% 
  bidirectional(
    layer_lstm(units = 32)
  ) %>% 
  layer_dense(units = 1, activation = "sigmoid") %>% 
  compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = "acc"
  )

history <- model_imdb_bidirectional %>% 
  fit(
    x_train,
    y_train,
    epochs = 10,
    batch_size = 128,
    validation_split = 0.2
  )
  




