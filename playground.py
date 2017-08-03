import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

images = []
measurements = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # Skip column names

    for row in reader:
        correction = 0.2
        steering_center = float(row[3])
        steering_left = steering_center + correction    # Turn right
        steering_right = steering_center - correction   # Turn left

        steering_center_flip = -steering_center
        steering_left_flip = -steering_left
        steering_right_flip = -steering_right

        directory = './data/'

        img_center = cv2.imread(directory + row[0].lstrip())
        img_left = cv2.imread(directory + row[1].lstrip())
        img_right = cv2.imread(directory + row[2].lstrip())

        img_center_flip = np.fliplr(img_center)
        img_left_flip = np.fliplr(img_left)
        img_right_flip = np.fliplr(img_right)

        images.extend([
            img_center,
            img_center_flip,
            img_left,
            img_left_flip,
            img_right,
            img_right_flip])
        measurements.extend([
            steering_center,
            steering_center_flip,
            steering_left,
            steering_left_flip,
            steering_right,
            steering_right_flip])


X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()

model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Conv2D(6, (5,5), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(6, (5,5), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5, verbose=1)

model.save('model.h5')

# Augment data with recovery laps showing only corrections
# Build NVIDIA pipeline
# Record extra laps
    # 2-3 good laps with center driving
    # 1 recovery lap
    # 1 lap very smooth in curves
    # ? in opposite direction
# Can visualize history with pyplot graphs
# Use generators to load data on demand
# Record video of 1+ autonomous laps
