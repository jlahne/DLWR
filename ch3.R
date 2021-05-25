# Chapter 3
library(keras)

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

