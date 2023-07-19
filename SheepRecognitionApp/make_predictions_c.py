# %%
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("sheep_classifier_model2.h5")

# Set the image size
img_width, img_height = 150, 150

# Initialize a list to store the selected image paths
selected_images = []
current_index = -1  # Initialize the current index as -1
displayed_images = []  # Initialize a list to store the previously displayed images


# Function to predict the class of the selected image
def predict_class():
    if len(selected_images) > 0:
        # If images exist, display the first image and update the predicted class
        current_index = 0
        display_image(selected_images[current_index])
        update_predicted_class()


# Function to display the selected image
def display_image(image_path):
    image = Image.open(image_path)
    image.thumbnail((400, 400))  # Resize the image to fit the display
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo

    # Center the image on the window
    image_label.pack(pady=10)

    # Add the displayed image to the list
    displayed_images.append(image_path)


# Function to display the next image in the list
def next_image():
    global current_index
    if current_index < len(selected_images) - 1:
        current_index += 1
    elif current_index == len(selected_images) - 1 and len(displayed_images) > 0:
        current_index = 0
    display_image(selected_images[current_index])
    update_predicted_class()


# Function to display the previous image in the list
def previous_image():
    global current_index
    if current_index > 0:
        current_index -= 1
    elif current_index == 0 and len(displayed_images) > 0:
        current_index = len(selected_images) - 1
    display_image(selected_images[current_index])
    update_predicted_class()


# Function to update the predicted class
def update_predicted_class():
    image_path = selected_images[current_index]
    image = load_img(image_path, target_size=(img_width, img_height))
    image_array = img_to_array(image) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_batch)
    predicted_class = "sheep" if predictions[0] >= 0.5 else "non-sheep"
    predicted_label.config(text=f"Predicted class: {predicted_class}")


# Function to select a directory
def select_directory():
    global selected_images  # Add a global statement to update the variable
    # Select the directory containing images
    dir_path = filedialog.askdirectory(initialdir="/", title="Select Directory")

    # Retrieve all image file paths from the selected directory
    image_paths = [
        os.path.join(dir_path, filename)
        for filename in os.listdir(dir_path)
        if filename.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Update the selected image paths list
    selected_images = image_paths
    predict_class()


# Create the main application window
root = tk.Tk()
root.title("Sheep Recognition App")
root.geometry("600x600")  # Set the window size to 600x600 pixels
root.configure(bg="white")  # Set the background color to white

# Create and configure the image label
image_label = tk.Label(root)
image_label.configure(bg="white")

# Create and configure the predicted label
predicted_label = tk.Label(root, font=("Arial", 14))
predicted_label.configure(bg="white")
predicted_label.pack(pady=10)

# Create and configure the Next and Previous buttons
previous_button = tk.Button(
    root, text="<", font=("Arial", 16), bd=0, command=previous_image
)
previous_button.configure(relief=tk.RAISED)
previous_button.place(relx=0.1, rely=0.5, anchor=tk.CENTER)

next_button = tk.Button(root, text=">", font=("Arial", 16), bd=0, command=next_image)
next_button.configure(relief=tk.RAISED)
next_button.place(relx=0.9, rely=0.5, anchor=tk.CENTER)

# Create and configure the Select Directory button
select_button = tk.Button(root, text="Select Directory", command=select_directory)
select_button.configure(relief=tk.RAISED)
select_button.pack(side=tk.BOTTOM, pady=10)

# Start the main application loop
root.mainloop()
# %%
