import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open('./data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  next(reader)
  for line in reader:
    lines.append(line)

images = []
measurements = []
for line in lines:
  source_path = line[0]
  filename = source_path.split('/')[-1]
  current_path = './data/IMG/' + filename
  image = cv2.imread(current_path)
  images.append(image)
  measurement = float(line[3])
  measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()

model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320,3)))
model.add(Conv2D(6, (5,5), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D((6, (5,5), activation='relu')))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5, verbose=1)

model.save('model.h5')

# Add normalization and mean centering using a Lambda
# Augment data by flipping images with inverse values
# Augment data with recovery laps showing only corrections
# Build NVIDIA pipeline
# Consider using other camera images
# Crop and scale down images
# Record extra laps
    # 2-3 good laps with center driving
    # 1 recovery lap
    # 1 lap very smooth in curves
    # ? in opposite direction
# Can visualize history with pyplot graphs
# Use generators to load data on demand
# Record video of 1+ autonomous laps
