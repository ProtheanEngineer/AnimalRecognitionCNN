# main.py
from data_preprocessing import prepare_dataset
from model_definition import create_model

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam, SGD

# Set your parameters
data_directory = 'dataset'
img_height = 255
img_width = 255
batch_size = 32
num_classes = 10
num_epochs = 15

# Prepare dataset
train_generator, validation_generator = prepare_dataset(data_directory, img_height, img_width, batch_size)

# Create model
model = create_model(img_height, img_width, num_classes)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['binary_accuracy', 'categorical_accuracy'])

# Train the model
history = model.fit(train_generator, epochs=num_epochs, validation_data=validation_generator)

# Save the model
model.save('animal_recognition_model.h5')


# Evaluate model
# evaluate_model(model, test_generator)  # Assuming you have a test_generator

# Make predictions
# predictions = make_predictions(model, new_images)
