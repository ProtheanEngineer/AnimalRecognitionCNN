from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the trained model
model_path = 'animal_recognition_model.h5'  # Replace with the path to your saved model
model = load_model(model_path)

# Load the test image
img_path = 'test/dog.jpg'  # Replace with the path to your test image
img = image.load_img(img_path, target_size=(150, 150))

# Define the path to your dataset folder
dataset_folder = 'dataset'

# Get the class names from the subdirectories in the dataset folder
class_names = sorted(os.listdir(dataset_folder))

# Convert the image to a numpy array and preprocess it
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make a prediction
predictions = model.predict(img_array)

# Get the index of the predicted class
predicted_class_index = np.argmax(predictions[0])

# Retrieve the class name for the predicted class index
predicted_class = class_names[predicted_class_index]

print(f"Predicted class: {predicted_class}")



# Display the original image
plt.imshow(img)
plt.axis('off')
plt.show()
