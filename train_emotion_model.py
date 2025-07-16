import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# Simulated FER-2013 (since real FER dataset requires download)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = tf.image.rgb_to_grayscale(tf.image.resize(X_train, [48, 48]))[:7000]
X_test = tf.image.rgb_to_grayscale(tf.image.resize(X_test, [48, 48]))[:1400]
y_train = y_train[:7000] % 7
y_test = y_test[:1400] % 7

X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = to_categorical(y_train, 7)
y_test = to_categorical(y_test, 7)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=64)

model.save("emotion_model.h5")
print("Model saved as emotion_model.h5")
