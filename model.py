from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import tensorflow as tf

# Set image size and batch size
img_height, img_width = 256, 256
batch_size = 32

# Augmentation for training, basic rescaling for testing
datagen_train = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.15,
                                   zoom_range=0.15,
                                   horizontal_flip=True)
datagen_test = ImageDataGenerator(rescale=1./255)

# Load datasets
train_ds = datagen_train.flow_from_directory(
    'images/Training',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

test_ds = datagen_test.flow_from_directory(
    'images/Testing',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

# Define CNN model
classifier = Sequential()

# Convolution and Pooling layers
classifier.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening layer
classifier.add(Flatten())

# Dense layers
classifier.add(Dense(units=512, activation='relu'))
classifier.add(BatchNormalization())
classifier.add(Dense(256, activation='relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(units=2, activation='softmax'))

# Compile model
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = classifier.fit(train_ds, epochs=10)

# Save the trained model
model_path = 'model.h5'  # Specify the path where you want to save the model
classifier.save(model_path)
print(f"Model saved to {model_path}")

# Evaluate the model on test data
test_loss, test_accuracy = classifier.evaluate(test_ds)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
