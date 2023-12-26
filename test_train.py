# Import necessary libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Replace with the path to your trained model
model_path = "animal_recognition_model.h5"

# Load the trained model
model = load_model(model_path)

# Replace with the path to your training data directory
train_data_directory = "resized_train"

# Define image data generator for training data
train_datagen = ImageDataGenerator(rescale=1./255)  # You may need to adjust other parameters based on your preprocessing

# Create a generator for training data
train_generator = train_datagen.flow_from_directory(
    train_data_directory,
    target_size=(255, 255),
    batch_size=32,  # Adjust the batch size based on your needs
    class_mode='categorical'  # Assuming you have categorical labels
)

# Evaluate the model on the training data
train_evaluation = model.evaluate(train_generator)
print("Training Evaluation:", train_evaluation)
