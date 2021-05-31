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
  













