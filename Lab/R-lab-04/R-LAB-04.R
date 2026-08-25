install.packages("Rcpp")
install.packages("keras")
install.packages("BiocManager")
BiocManager::install("EBImage")

# Load Packages
library(EBImage)
library(keras)

# Read images
setwd('C:/Users/~katkar/OneDrive - Vidyalankar Institute of Technology/Desktop/R-PROGRAMMING/Lab/R-lab-04/Rlab4')

pics <- c('p1.jpg', 'p2.jpg', 'p3.jpg', 'p4.jpg', 'p5.jpg', 'p6.jpg',
          'c1.jpg', 'c2.jpg', 'c3.jpg', 'c4.jpg', 'c5.jpg', 'c6.jpg')

mypic <- list()

for (i in 1:12) {
  mypic[[i]] <- readImage(pics[i])
}


# Explore
print(mypic[[1]])
display(mypic[[8]])
summary(mypic[[1]])
hist(mypic[[2]])
str(mypic)


# Resize
for (i in 1:12) {
  mypic[[i]] <- resize(mypic[[i]], 28, 28)
}


# Reshape / Flatten
for (i in 1:12) {
  mypic[[i]] <- matrix(as.vector(mypic[[i]]), nrow = 1)
}


# Check dimensions
dim(mypic[[1]])


# Row Bind
trainx <- NULL

for (i in 7:11) {
  trainx <- rbind(trainx, mypic[[i]])
}

str(trainx)


# Test data
testx <- rbind(mypic[[6]], mypic[[12]])


# Labels
trainy <- c(0,0,0,0,0,1,1,1,1,1)
testy <- c(0,1)


# One Hot Encoding
trainLabels <- matrix(0, nrow = length(trainy), ncol = 2)
trainLabels[cbind(1:length(trainy), trainy + 1)] <- 1

testLabels <- matrix(0, nrow = length(testy), ncol = 2)
testLabels[cbind(1:length(testy), testy + 1)] <- 1


# Check labels
trainLabels
testLabels


# Model
model <- keras_model_sequential()

model %>%
  layer_dense(
    units = 256,
    activation = 'relu',
    input_shape = c(2352)
  ) %>%
  layer_dense(
    units = 128,
    activation = 'relu'
  ) %>%
  layer_dense(
    units = 2,
    activation = 'softmax'
  )

summary(model)


# Compile
model %>%
  compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
  )


# Fit Model
history <- model %>%
  fit(
    trainx,
    trainLabels,
    epochs = 30,
    batch_size = 32,
    validation_split = 0.2
  )


# Evaluation & Prediction - Train Data
model %>%
  evaluate(trainx, trainLabels)


pred <- model %>%
  predict_classes(trainx)

table(
  Predicted = pred,
  Actual = trainy
)


prob <- model %>%
  predict_proba(trainx)

cbind(
  prob,
  Predicted = pred,
  Actual = trainy
)