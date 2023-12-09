# data_preprocessing.py
import os
import shutil

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def prepare_dataset(data_directory, img_height, img_width, batch_size):
    # Create train and validation folders if not present
    train_folder = os.path.join('train', 'train')  # Change this path
    validation_folder = os.path.join('validation', 'validation')  # Change this path

    for folder in [train_folder, validation_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Move images to train and validation folders
    for animal_class in os.listdir(data_directory):
        class_folder = os.path.join(data_directory, animal_class)
        if os.path.isdir(class_folder):
            images = os.listdir(class_folder)
            num_images = len(images)
            num_train = int(0.8 * num_images)  # 80% for training

            train_images = images[:num_train]
            validation_images = images[num_train:]

            for img in train_images:
                src = os.path.join(class_folder, img)
                dst = os.path.join(train_folder, animal_class, img)

                # Ensure the destination directory exists before copying
                os.makedirs(os.path.dirname(dst), exist_ok=True)

                shutil.copy(src, dst)

            for img in validation_images:
                src = os.path.join(class_folder, img)
                dst = os.path.join(validation_folder, animal_class, img)

                # Ensure the destination directory exists before copying
                os.makedirs(os.path.dirname(dst), exist_ok=True)

                shutil.copy(src, dst)

    # Create ImageDataGenerators for train and validation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        train_folder,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = validation_datagen.flow_from_directory(
        validation_folder,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, validation_generator
