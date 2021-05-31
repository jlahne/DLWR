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

# Pretrained models (VGG16) - restart R here

rm(list = ls()); gc()
library(keras)
library(tidyverse)

conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

train_dir <- "~/Documents/R/Deep Learning with R/cats_and_dogs_small/train"
validation_dir <- "~/Documents/R/Deep Learning with R/cats_and_dogs_small/validation"
test_dir <- "~/Documents/R/Deep Learning with R/cats_and_dogs_small/test"

train_cats_dir <- file.path(train_dir, "cats")
train_dogs_dir <- file.path(train_dir, "dogs")
test_cats_dir <- file.path(test_dir, "cats")
test_dogs_dir <- file.path(test_dir, "dogs")
validation_cats_dir <- file.path(test_dir, "cats")
validation_dogs_dir <- file.path(test_dir, "dogs")

extract_features <- function(directory, sample_count){

  datagen <- image_data_generator(rescale = 1 / 255)
  batch_size <- 20
  
  features <- array(0, dim = c(sample_count, 4, 4, 512))
  labels <- array(0, dim = c(sample_count))
  
  generator <- flow_images_from_directory(
    directory = directory,
    generator = datagen,
    target_size = c(150, 150),
    batch_size = batch_size,
    class_mode = "binary"
  )
  
  i <- 0
  while(TRUE){
    batch <- generator_next(generator)
    inputs_batch <- batch[[1]]
    labels_batch <- batch[[2]]
    features_batch <- conv_base %>% predict(inputs_batch)
    
    index_range <- ((i * batch_size) + 1):((i + 1) * batch_size)
    features[index_range, , , ] <- features_batch
    labels[index_range] <- labels_batch
    
    i <- i + 1
    if(i * batch_size >= sample_count)
      break
  }
  
  list(features = features,
       labels = labels)
  
}

train <- extract_features(train_dir, 2000)
validation <- extract_features(validation_dir, 1000)
test <- extract_features(test_dir, 1000)

reshape_features <- function(features){
  array_reshape(features, dim = c(nrow(features), 4 * 4 * 512))
}

train$features <- reshape_features(train$features)
validation$features <- reshape_features(validation$features)
test$features <- reshape_features(test$features)

model_cheapvgg <- keras_model_sequential() %>% 
  layer_dense(units = 256, activation = "relu", input_shape = 4 * 4 * 512) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid") %>% 
  compile(
    optimizer = optimizer_rmsprop(lr = 2e-5),
    loss = loss_binary_crossentropy,
    metrics = "acc"
  )

history <- model_cheapvgg %>% 
  fit(
    train$features,
    train$labels,
    epochs = 30,
    batch_size = 20,
    validation_data = list(validation$features, validation$labels)
  )

# The next section of the chapter requires methods that are computationally intractable without a GPU (apparently)
# restart R here

rm(list = ls()); gc()
library(keras)
library(tidyverse)

model <- load_model_hdf5("cats_and_dogs_small_2.h5")
model

img_path <- "~/Documents/R/Deep Learning with R/cats_and_dogs_small/test/cats/cat.1523.jpg"
img <- image_load(img_path, target_size = c(150, 150))
img_tensor <- image_to_array(img)
img_tensor <- array_reshape(img_tensor, c(1, 150, 150, 3))
img_tensor <- img_tensor / 255

dim(img_tensor)
plot(as.raster(img_tensor[1, , , ]))

layer_outputs <- lapply(model$layers[1:8], function(layer) layer$output)
activation_model <- keras_model(inputs = model$input, outputs = layer_outputs)

activations <- activation_model %>% 
  predict(img_tensor)

plot_channel <- function(channel){
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(channel), axes = FALSE, asp = 1, col = terrain.colors(12))
}

plot_channel(activations[[1]][1, , , 2])
plot_channel(activations[[1]][1, , , 7])

image_size <- 58
images_per_row <- 16

for(i in 1:8){
  
  layer_activation <- activations[[i]]
  layer_name <- model$layers[[i]]$name
  
  n_features <- dim(layer_activation)[[4]]
  n_cols <- n_features %/% images_per_row
  
  png(paste0("cat_activations_", i, "_", layer_name, ".png"),
      width = image_size * images_per_row,
      height = image_size * n_cols)
  op <- par(mfrow = c(n_cols, images_per_row),
            mai = rep_len(0.02, 4))
  
  for(col in 0:(n_cols - 1)){
    for(row in 0:(images_per_row - 1)){
      channel_image <- layer_activation[1, , , (col * images_per_row) + row + 1]
      plot_channel(channel_image)
    }
  }

  par(op)
  dev.off()
    
}

# Gradient ascent to visualize a particular filter

library(keras)
library(tidyverse)

#model <- load_model_hdf5("cats_and_dogs_small_2.h5")
#layer_name <- "conv2d_29"

model <- application_vgg16(weights = "imagenet", include_top = FALSE)
layer_name <- "block3_conv1"
filter_index <- 1



