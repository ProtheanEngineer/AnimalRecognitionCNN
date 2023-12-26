from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
import tensorflow

from PIL import Image

# Replace with the actual path to your image folder
image_folder = "test"

# Replace with the desired target size
target_size = (255, 255)

# Iterate through all images in the folder
for filename in os.listdir(image_folder):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(image_folder, filename)

        # Load the image
        img = Image.open(image_path)

        # Resize the image
        img_resized = img.resize(target_size, Image.LANCZOS)

        # Save the resized image, overwriting the original image
        img_resized.save(image_path)

# Load the trained model
model_path = 'animal_recognition_model.h5'  # Replace with the path to your saved model
model = load_model(model_path)

model.reset_states()


# Define the path to your dataset folder
dataset_folder = 'test'

# Get the class names from the subdirectories in the dataset folder
class_names = os.listdir(dataset_folder)
print(class_names)

# Load the test image
img_path = 'test/perro/dog3.jpg'  # Replace with the path to your test image
img = image.load_img(img_path, target_size=target_size)

# Convert the image to a numpy array and preprocess it
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make a prediction
predictions = model.predict(img_array)
print(predictions)

# Get the index of the predicted class
predicted_class_index = np.argmax(predictions[0])

# Retrieve the class name for the predicted class index
predicted_class = class_names[predicted_class_index]

print(f"Predicted class: {predicted_class}")
# print(img_array.shape)
# Display the original image
plt.imshow(img)
plt.axis('off')
plt.show()