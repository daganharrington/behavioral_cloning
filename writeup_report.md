# Introduction

This repository provides a neural network model intended to train on virtual camera images
from udacity's self-driving-car-simulator. The files contained here can be used to
train on data provided by the [self-driving-car-simulator](https://github.com/udacity/self-driving-car-sim)
and the trained model can then be used to emit heading information to the simulator while
running in autonomous mode.

[VIDEO](https://www.youtube.com/watch?v=Fk0ysUuZ9as)


The goals of this project are the following:
* Build up a driving dataset using the simulator. Care had to be taken to iterate over a decent model and
    append more behavioral data to dataset to express/cover pathological cases and problem spots in the
    evaluation of the model; that is, helping the model generalize to the whole track often involved
    finding in what circumstances the model performed poorly, and adding data to cover those cases.
* Build a convolutional neural network suitable for training on these images.
* Train and validate the model to produce a reasonable fit to the data.
    Many regularization techniques were tried (dropout, l2 normalization)
* Test that the model successfully drives around track one without leaving the road


# Files
  * model.py
    ** A convolutional neural network built with keras.
    ** A generator to emit batches of images
  * model.h5
    ** A trained network
  * drive.py
    ** Emits predicted heading information (given a trained model)
       to the simulator over a flask api.
  * writeup_report.md (you're looking at it)


# Usage
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

# Model Architecture and Training Strategy

## Model:

The model loosely follows the [nVidia architecture](https://arxiv.org/pdf/1604.07316.pdf).
A normalization plane using `keras.layers.Lambda` and `keras.layers.Cropping2D` applies
data normalization and image cropping to every input vector. 5 convolutional feature maps
then follow, with differing kernel sizes (3 5x5, 2 3x3) and a RELU for each to introduce
nonlinearities. Between 2 conv layers, I added some max pooling and found that results
generalized better.

After the convolutional layers, there is a collection of fully connected layers, reducing
down to our single node which is our steering angle prediction. This entire architecture
solves a regression problem for a 1d vector over the reals.

I also added dropout layers [because Hinton made me do it](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf).

```
    model = Sequential()

    model.add(Cropping2D(cropping=((70, 25), (0,0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))

    model.add(Dropout(0.2))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2),  activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1)))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))

    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    model.add(Flatten())

    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))

    model.add(Dense(1))
```

## Training/Learning

[ADAM](https://arxiv.org/abs/1412.6980v8) is used for gradient-based learning. This obviates
the need to tune a learning rate with some magic numbers.

## Data wrangling/gathering

The track is a counter-clockwise loop. This bias toward turning left was counteracted
by flipping each image across the vertical axis and flipping each heading value
across zero (-h). This had the glorious side-effect of doubling the amount of data.

It was also important to train the model around the knowledge of "how to get back on
the track from the edges". If all of the behavioral data was "perfect driving", the model
would likely be at a loss for how to correct when things go wrong. I spent some time around
troubled curves collecting data while driving back toward the center of the track. This helped
immensely.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 
The validation set gave a clear indication when the model was overfitting and that
perhaps the number of epochs during the training phase should be reduced.
