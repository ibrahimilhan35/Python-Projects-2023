#%%
"""
Python code snippet that uses the TensorFlow library to train a convolutional neural network (CNN) to classify sheep and non-sheep images from a directory:
"""
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the path to your dataset directory
dataset_dir = "/Users/GeoPhysicist/Python-Projects-2023/SheepRecognitionApp"


# Set the image size and batch size for training
img_width, img_height = 150, 150

batch_size = 32


# Create an ImageDataGenerator for data augmentation and preprocessing
datagen = ImageDataGenerator(rescale=1.0/255.0)


# Load and prepare the training data
train_data = datagen.flow_from_directory(
    os.path.join(dataset_dir, 'train'),
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    classes=['non_sheep', 'sheep']
)

# Load and prepare the validation data
val_data = datagen.flow_from_directory(
    os.path.join(dataset_dir, 'val'),
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    classes=['non_sheep', 'sheep']
)


# Build the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(
    train_data,
    steps_per_epoch=train_data.samples // batch_size,
    epochs=10,
    validation_data=val_data,
    validation_steps=val_data.samples // batch_size
)


# Save the trained model
model.save('sheep_classifier_model.h5')


#%%
