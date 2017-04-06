import tensorflow as tf

from csv import DictReader
from keras.layers.core import Dropout
from keras.layers import Convolution2D, Cropping2D, Dense, Flatten, Lambda
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from numpy import array, flip
from os.path import join
from random import random
from scipy.ndimage import imread
from time import time

#wget https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip

DATA_DIRECTORIES = ['data', 'data_2', 'data_3', 'data_4', 'data_5', 'data_6', 'data_7', 'data_8']
LOG_FILE = 'driving_log.csv'
MODEL_DIR = 'model_2017_04_05_9PM'
MODEL_NAME = 'model.h5'

def load_training_data(p_data_directories):
    images = []
    steerings = []

    for data_directory in p_data_directories:
        log_file_path = join(data_directory, LOG_FILE)
        with tf.gfile.GFile(log_file_path, mode='r') as log_file:
            for row in DictReader(log_file):
                left_image = imread(join(data_directory, row['left']))
                center_image = imread(join(data_directory, row['center']))
                right_image = imread(join(data_directory, row['right']))

                steering = float(row['steering'])

                # Down sampling no steering
                if (steering == 0) and (random() < 0.8):
                    images.append(center_image)
                    steerings.append(steering)

                images.append(left_image)
                steerings.append(steering + 0.3)
                images.append(right_image)
                steerings.append(steering - 0.3)

                images.append(flip(center_image, axis=1))
                steerings.append(steering * -1.0)

    return array(images)), array(steerings


def define_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x * 2.0 - 1.0))
    model.add(Convolution2D(4, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(8, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    # Drop out now useful since the model size is small enough
    # model.add(Dropout(0.1))
    model.add(Dense(64))
    model.add(Dense(1))

    return model

def train_model(model, x, y):
    model.compile(loss='mse', optimizer='adam')
    # Higer batch size to speed up training
    model.fit(x, y, validation_split=0.05, shuffle=True, nb_epoch=5, batch_size=128)

def main():
    main_start_time = time()

    images, steerings = load_training_data(DATA_DIRECTORIES)
    model = define_model()
    train_model(model, images, steerings)

    model.save(join(MODEL_DIR, MODEL_NAME))

    print('Done in %d seconds' % (time() - main_start_time))

if __name__ == "__main__":
    main()