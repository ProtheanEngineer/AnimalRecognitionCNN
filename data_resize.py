from PIL import Image
import os

# Path to your train and validation folders
train_folder = "train/train"
validation_folder = "validation/validation"

# Output folders for resized images
resized_train_folder = "resized_train"
resized_validation_folder = "resized_validation"

# Target size
target_size = (255, 255)

# Resize images in the train folder
for class_folder in os.listdir(train_folder):
    class_input_folder = os.path.join(train_folder, class_folder)
    class_output_folder = os.path.join(resized_train_folder, class_folder)
    os.makedirs(class_output_folder, exist_ok=True)

    for img_name in os.listdir(class_input_folder):
        img_path = os.path.join(class_input_folder, img_name)
        img = Image.open(img_path)
        img_resized = img.resize(target_size, Image.LANCZOS)
        img_resized.save(os.path.join(class_output_folder, img_name))

# Resize images in the validation folder
for class_folder in os.listdir(validation_folder):
    class_input_folder = os.path.join(validation_folder, class_folder)
    class_output_folder = os.path.join(resized_validation_folder, class_folder)
    os.makedirs(class_output_folder, exist_ok=True)

    for img_name in os.listdir(class_input_folder):
        img_path = os.path.join(class_input_folder, img_name)
        img = Image.open(img_path)
        img_resized = img.resize(target_size, Image.LANCZOS)
        img_resized.save(os.path.join(class_output_folder, img_name))
