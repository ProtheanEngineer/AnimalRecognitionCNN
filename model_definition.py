from tensorflow.keras import layers, models

def create_model(img_height, img_width, num_classes):
    # Codigo de antes (75%)
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.5))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.5))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.5))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.5))
    
    # Codigo nuevo (layer 256)
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    # model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    
    # model=models.Sequential()
    
    # model.add(layers.Conv2D(32, kernel_size = 3, activation='relu', input_shape = (img_height, img_width, 3)))

    # model.add(layers.Conv2D(32, kernel_size = 3, activation='relu'))

    # model.add(layers.Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))

    # model.add(layers.Dropout(0.4))
    # model.add(layers.Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))

    # model.add(layers.Dropout(0.4))
    # model.add(layers.Conv2D(256, kernel_size = 4, activation='relu'))

    # model.add(layers.Flatten())
    # model.add(layers.Dropout(0.4))
    # model.add(layers.Dense(num_classes, activation='softmax'))

    return model
