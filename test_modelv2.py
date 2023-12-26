import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load your trained model
# Replace 'your_model.h5' with the actual filename of your trained model
model = load_model('animal_recognition_model.h5')

# Path to the 'test' folder
test_folder = 'test'

# Initialize lists to store true and predicted labels
true_labels = []
predicted_labels = []

# Get a list of class folders within the 'test' folder
class_folders = [class_folder.path for class_folder in os.scandir(test_folder) if class_folder.is_dir()]

# Create a mapping of class indices to class names
class_names = [os.path.basename(class_folder) for class_folder in class_folders]

# Create an ImageDataGenerator with normalization and augmentations
datagen = image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Iterate through each class folder
for class_folder in class_folders:
    # Get a list of image files in the class folder
    image_files = [f.path for f in os.scandir(class_folder) if f.is_file() and f.name.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Get the class label (assuming the folder name is the class label)
    class_label = os.path.basename(class_folder)

    # Iterate through each image in the class folder
    for image_path in image_files:
        # Load, rescale, and preprocess the test image
        img = image.load_img(image_path, target_size=(255, 255))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make predictions using the model
        predictions = model.predict(img_array)

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)

        # Map the predicted class index to the class name
        predicted_class_name = class_names[predicted_class_index]

        # Display the image filename, true class, and predicted class
        print(f"Image: {os.path.basename(image_path)}, True Class: {class_label}, Predicted Class: {predicted_class_name}")

        # Append true and predicted labels for later analysis
        true_labels.append(class_label)
        predicted_labels.append(predicted_class_name)
        
        # print(predictions.shape)
        # print(predictions)