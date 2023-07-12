# %%
"""
This part makes predictions of images from a directory
"""
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("sheep_classifier_model.h5")

# Set the path to the new image directory
new_image_dir = "/Users/GeoPhysicist/Python-Projects-2023/SheepRecognitionApp/test"

# Set the image size
img_width, img_height = 150, 150

# Iterate over the new images
for filename in os.listdir(new_image_dir):
    image_path = os.path.join(new_image_dir, filename)
    image = load_img(image_path, target_size=(img_width, img_height))
    image_array = img_to_array(image) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)

    # Make predictions on the image batch
    predictions = model.predict(image_batch)
    predicted_class = "sheep" if predictions[0] >= 0.5 else "non-sheep"

    print(f"Image: {filename}, Predicted class: {predicted_class}")


#%%