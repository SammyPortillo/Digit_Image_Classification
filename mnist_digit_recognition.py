# Simple neural network for classifying MNIST
# digits using TensorFlow and Keras.
# Evaluates the performance of the model using a confusion matrix.

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Reshape input data
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Show first 10
fig, ax = plt.subplots(nrows=1, ncols=10)
for i in range(10):
    ax[i].imshow(train_images[i], cmap='gray')
    ax[i].title.set_text(train_labels[i])
plt.show()


# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=2)

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels)
print('Loss:', loss)
print('Accuracy:', acc)

# Predict the labels for test images
predictions = np.argmax(model.predict(test_images), axis=-1)

# Create a confusion matrix
conf_mat = confusion_matrix(test_labels, predictions)

# Display the confusion matrix using seaborn heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
