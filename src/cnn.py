import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from sklearn.model_selection import train_test_split

from data import load_train

K.set_image_dim_ordering('th')

seed = 7
numpy.random.seed(seed)
sample_size = 20000

train_images, train_labels = load_train()
train_images = train_images[:sample_size]
train_labels = train_labels[:sample_size]
X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.3, random_state=42)

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 64, 64).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 64, 64).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def larger_model():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 64, 64), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = larger_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=6, batch_size=128)
scores = model.evaluate(X_test, y_test, verbose=0)

print(f'Large CNN Error: {100-scores[1]*100:.2f}')
