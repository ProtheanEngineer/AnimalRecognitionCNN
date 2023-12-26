import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

# Load your trained model
# Replace 'your_model.h5' with the actual filename of your trained model
model = load_model('animal_recognition_model.h5')

# Path to the 'test' folder
test_folder = 'test'

# Initialize lists to store true and predicted labels
true_labels = []
predicted_labels = []

# Create an ImageDataGenerator with normalization and augmentations
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Create a generator for the 'test' directory
test_generator = datagen.flow_from_directory(
    test_folder,
    target_size=(255, 255),
    batch_size=1,
    class_mode='categorical',
    shuffle=False  # Important to keep the order
)

# Get class names
class_names = list(test_generator.class_indices.keys())

# Make predictions using the model
predictions = model.predict(test_generator)

# Iterate through each image in the generator
for i in range(len(test_generator.filenames)):
    # Get the true class index
    true_class_index = test_generator.classes[i]

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions[i])

    # Map the true and predicted class indices to class names
    true_class_name = class_names[true_class_index]
    predicted_class_name = class_names[predicted_class_index]

    # Display the image filename, true class, and predicted class
    print(f"Image: {test_generator.filenames[i]}, True Class: {true_class_name}, Predicted Class: {predicted_class_name}")
    
    # Load the image
    img_path = os.path.join(test_folder, test_generator.filenames[i])
    img = load_img(img_path, target_size=(255, 255))

    # Display the image with the correct and predicted classes in the title
    plt.imshow(img)
    plt.title(f'True Class: {true_class_name}\nPredicted Class: {predicted_class_name}')
    plt.show()