rm(list = ls()); gc()
library(keras)
library(tidyverse)

model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(28, 28, 1)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu")

model <- model %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")

mnist <- dataset_mnist()
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

train_images <- array_reshape(train_images, c(60000, 28, 28, 1))
train_images <- train_images / 255
test_images <- array_reshape(test_images, c(10000, 28, 28, 1))
test_images <- test_images / 255

train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

model %>% 
  compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = "accuracy"
  )

model %>% 
  fit(
    train_images,
    train_labels,
    epochs = 5,
    batch_size = 64
  )

results <- model %>%
  evaluate(
    test_images,
    test_labels
  )

# Compare the model above (with maxpooling) to a model with stride = 2 in both dimensions - this is slightly faster?

stride_model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), padding = "same", strides = c(2, 2), activation = "relu", input_shape = c(28, 28, 1)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same", strides = c(2, 2), activation = "relu") %>% 
  layer_conv_2d(filter = 64, kernel_size = c(3, 3), padding = "same", strides = c(2, 2), activation = "relu") %>% 
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

stride_model %>%
  compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = "accuracy"
  )

stride_model %>%
  fit(
    train_images,
    train_labels,
    epochs = 5,
    batch_size = 64
  )

stride_model %>% evaluate(test_images, test_labels)

# Cats and dogs model

train_dir <- "~/Documents/R/Deep Learning with R/cats_and_dogs_small/train"
validation_dir <- "~/Documents/R/Deep Learning with R/cats_and_dogs_small/validation"
test_dir <- "~/Documents/R/Deep Learning with R/cats_and_dogs_small/test"

train_cats_dir <- file.path(train_dir, "cats")
train_dogs_dir <- file.path(train_dir, "dogs")
test_cats_dir <- file.path(test_dir, "cats")
test_dogs_dir <- file.path(test_dir, "dogs")
validation_cats_dir <- file.path(test_dir, "cats")
validation_dogs_dir <- file.path(test_dir, "dogs")

catdog_model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(150, 150, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 512, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

catdog_model %>%
  compile(
    optimizer = optimizer_rmsprop(lr = 1e-4),
    loss = loss_binary_crossentropy,
    metrics = "accuracy"
  )

train_datagen <- image_data_generator(rescale = 1 / 255)
validation_datagen <- image_data_generator(rescale = 1 / 255)

train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

batch <- generator_next(train_generator)
str(batch)

history <- catdog_model %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 100,
    epochs = 30,
    validation_data = validation_generator,
    validation_steps = 50
  )

save_model_hdf5(catdog_model, "cats_and_dogs_small_1.h5")

# Training with data augmentation

catdog_model_2 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(150, 150, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 512, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid") %>% 
  compile(
    optimizer = optimizer_rmsprop(lr = 1e-4),
    loss = loss_binary_crossentropy,
    metrics = "acc"
  )

augment_datagen <- image_data_generator(
  rescale = 1 / 255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

test_datagen <- image_data_generator(rescale = 1 / 255)

train_generator <- flow_images_from_directory(
  directory = train_dir, 
  generator = augment_datagen, 
  target_size = c(150, 150), 
  class_mode = "binary", 
  batch_size = 32
)

validation_generator <- flow_images_from_directory(
  directory = validation_dir,
  generator = test_datagen,
  target_size = c(150, 150),
  class_mode = "binary",
  batch_size = 32
)

history <- catdog_model_2 %>% 
  fit_generator(
    train_generator,
    steps_per_epoch = 62,
    epochs = 5,
    validation_data = validation_generator,
    validation_steps = 15
  )

