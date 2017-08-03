import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            images = []
            angles = []
            for sample in samples[offset:offset + batch_size]:
                directory = './data/'
                img_center = cv2.imread(directory + sample[0].lstrip())
                img_left = cv2.imread(directory + sample[1].lstrip())
                img_right = cv2.imread(directory + sample[2].lstrip())
                images.extend([
                    img_center,
                    np.fliplr(img_center),
                    img_left,
                    np.fliplr(img_left),
                    img_right,
                    np.fliplr(img_right)
                ])

                correction = 0.2
                angle_center = float(sample[3])
                angle_left = angle_center + correction      # go right
                angle_right = angle_center - correction     # go left
                angles.extend([
                    angle_center,
                    -angle_center,
                    angle_left,
                    -angle_left,
                    angle_right,
                    -angle_right
                ])

                yield sklearn.utils.shuffle(np.array(images), np.array(angles))


# Fetch CSV data rows
rows = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        rows.append(line)


# Split out training and validation data
train_samples, validation_samples = train_test_split(rows, test_size=0.2)


# Create data loading generators
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# Build NVIDIA network
model = Sequential()
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(35, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


# Compile and train model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(
    train_generator,
    steps_per_epoch = len(train_samples),
    validation_data = validation_generator,
    validation_steps = len(validation_samples),
    epochs = 5,
    verbose = 1
)


# Visualize MSE loss across epochs

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model MSE Loss')
plt.ylabel('Mean Squared Error (MSE) Loss')
plt.xlabel('Epoch')
plt.legend(['Training Set', 'Validation Set'], loc='upper right')
plt.show()


# Save the model
model.save('model.h5')


#   Next Steps:
#
#      Train model on AWS as is
#           Sanity check that car kind of works
#           Make note of epoch performance
#
#      Generate additional data
#           2-3 laps regular
#           1 lap of recovery
#           1 lap of curves
#
#           Then optional:
#               1-2 laps regular on second track
#               1 lap of recovery on second track
#               If model is not good, can try to adjust brightness
#
#      Train model for keeps
#           Ensure model is better than before
#
#      Verify autonomous mode works well
#
#      Create video during autonomous mode
#           python drive.py model.h5 run1
#           python video.py run1 (optional --fps 48)


# Overfitting = low training MSE, high validation MSE => collect more data
# Underfitting = high training MSE, high validation MSE
# If both MSEs are low but the car still is not functional
#   Determine where it happens, e.g. curves
#   Then collect more data for that situation