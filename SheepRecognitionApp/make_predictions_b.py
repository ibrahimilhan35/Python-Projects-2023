# %%
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("sheep_classifier_model.h5")

# Set the image size
img_width, img_height = 150, 150


# Function to predict the class of the selected image
def predict_class():
    # Select the image from the directory
    image_path = filedialog.askopenfilename(
        initialdir="/",
        title="Select Image",
        filetypes=(("Image Files", "*.jpg *.jpeg *.png"), ("All Files", "*.*")),
    )

    # Load and preprocess the image
    image = load_img(image_path, target_size=(img_width, img_height))
    image_array = img_to_array(image) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)

    # Make predictions on the image batch
    predictions = model.predict(image_batch)
    predicted_class = "sheep" if predictions[0] >= 0.5 else "non-sheep"

    # Display the image and predicted class
    display_image(image_path)
    predicted_label.config(text=f"Predicted class: {predicted_class}")


# Function to display the selected image
def display_image(image_path):
    image = Image.open(image_path)
    image.thumbnail((400, 400))  # Resize the image to fit the display
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo


# Create the main application window
root = tk.Tk()
root.title("Sheep Recognition App")
root.configure(bg="dark gray")

# Create and configure the image label
image_label = tk.Label(
    root, text="Click the Select Image button to unlock the world of woolly wonders!"
)
# image_label.configure(bg="dark gray")
image_label.pack(pady=10)


# Create and configure the predicted label
predicted_label = tk.Label(root)
predicted_label.configure(bg="dark gray")
predicted_label.pack(pady=10)


# Create and configure the Select Image button
select_button = tk.Button(root, text="Select Image", command=predict_class)
select_button.pack(pady=10)

# Start the main application loop
root.mainloop()


# %%
