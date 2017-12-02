import csv
import cv2
import itertools
import pdb
import numpy as np
import random
import sklearn

import keras
from keras.models import Sequential
from keras.layers import Cropping2D, Lambda
from keras.layers.core import (
    Dense,
    Activation,
    Flatten,
    Dropout
)
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization

from sklearn.model_selection import train_test_split


DRIVING_LOG_CSV = 'round4/driving_log.csv'


class BatchGenerator:
    """A emitter of image batches."""

    # Base location path of images saved in `DRIVING_LOG_CSV`
    img_template = 'round4/IMG/{}'

    def __init__(self, batch_size=32):
        self.batch_size = batch_size


    def emit(self, samples):
        """Generator to yield arrays of size `self.batch_size` * 2.

        Args:
           samples (list(str)) Lines from 'driving_log.csv'

        Returns:
            sequence of indexable data-structures
                Sequence of shuffled views of the collections. The original arrays are not impacted.
        """
        for chunk in itertools.cycle(self._batch(samples, self.batch_size)):
            images = []
            headings = []

            for line in chunk:
                if line is None:
                    continue

                center_img = self._get_image_from_path(line[1])
                center_img = cv2.cvtColor(center_img, cv2.COLOR_BGR2RGB)

                center_heading = float(line[3])

                # data is heavily biased toward straight driving.
                # keep those samples with prob 1/(2^3)
                # discard the rest
                #if (center_heading > -0.05 and
                #    center_heading < 0.05 and
                #    bool(random.getrandbits(3))):

                images.append(center_img)
                headings.append(center_heading)

                # Add flipped images and heading to balance out the
                # counter-clockwise path along track.
                images.append(np.fliplr(center_img))
                headings.append(center_heading*-1.0)

            yield sklearn.utils.shuffle(
                np.array(images), np.array(headings))


    def _batch(self, iterable, n=100):
        """Return chunks of size `n` from `iterable`."""
        return itertools.zip_longest(*[iter(iterable)]*n)


    def _get_image_from_path(self, f):
        splt = f.split('/')
        img_path = splt[-1]
        return cv2.imread(self.img_template.format(img_path))




def get_model():
    """A rough approximation of the nvidia architecture.

    Also including dropout, pooling, image cropping, image normalization,
    and weight normalization.
    """

    model = Sequential()

    #model.add(BatchNormalization(input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0,0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))

    model.add(Dropout(0.2))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2),  activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1)))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1)))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1)))

    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    model.add(Flatten())

    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))

    model.add(Dense(1))

    return model


def main():
    # Load features and labels.
    samples = []
    with open(DRIVING_LOG_CSV) as fh:
        for line in csv.reader(fh):
            samples.append(line)
    train_set, valid_set = train_test_split(samples, test_size=0.2)

    # A generator to emit feature/label pairs by size of `batch_size`.
    # Note: `train_gen` is "larger" than `train_set`, because BatchGenerator.emit
    # also includes horizontally-flipped images (and respective flipped headings).
    train_gen = BatchGenerator(batch_size=32).emit(train_set)
    valid_gen = BatchGenerator(batch_size=32).emit(valid_set)


    model = get_model()

    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_gen,
                        samples_per_epoch=len(train_set)*2, # generators output the flipped images and headings, too
                        validation_data=valid_gen,
                        nb_val_samples=len(valid_set)*2,
                        nb_epoch=3)

    model.save('model.h5')



if __name__ == '__main__':
    main()
