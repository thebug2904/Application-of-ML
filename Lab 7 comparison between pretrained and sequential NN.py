import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define sequential model
model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

# Evaluate the model
_, acc = model.evaluate(x_test, y_test)
print("Sequential Model Accuracy: {:.2f}%".format(acc * 100))

# Load pre-trained ResNet50 model
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the layers
for layer in resnet_model.layers:
    layer.trainable = False

# Add custom classifier on top
resnet_top = Sequential([
    resnet_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
resnet_top.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
resnet_top.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

# Evaluate the model
_, resnet_acc = resnet_top.evaluate(x_test, y_test)
print("ResNet50 Transfer Learning Model Accuracy: {:.2f}%".format(resnet_acc * 100))

# Prediction comparison
y_pred = np.argmax(model.predict(x_test), axis=-1)
y_resnet_pred = np.argmax(resnet_top.predict(x_test), axis=-1)

print("Sequential Model Accuracy (Test Data): {:.2f}%".format(accuracy_score(np.argmax(y_test, axis=1), y_pred) * 100))
print("ResNet50 Transfer Learning Model Accuracy (Test Data): {:.2f}%".format(accuracy_score(np.argmax(y_test, axis=1), y_resnet_pred) * 100))
