import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import dotenv
import os
dotenv.load_dotenv()
from pathlib import Path

#set path variables:
train_path = os.environ['TRAIN_PATH']
train_path = Path("/mnt",train_path).as_posix()
test_path = os.environ['TEST_PATH']
test_path = Path(test_path).as_posix()


#check that gpu is being used
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  print('TensorFlow is using the GPU')
else:
  print('TensorFlow is not using the GPU')


# Step 1: Data Preparation
# Initialize ImageDataGenerator for data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Generate training dataset from directory with specified parameters
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(150, 150),  # Resize images to 150x150 pixels
    batch_size=32,  # Batch size for training
    class_mode='binary',  # Binary classification (normal or pneumonia)
    subset='training'  # Use training subset
)

# Generate validation dataset from directory with specified parameters
validation_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(150, 150),  # Resize images to 150x150 pixels
    batch_size=32,  # Batch size for validation
    class_mode='binary',  # Binary classification (normal or pneumonia)
    subset='validation'  # Use validation subset
)

# Step 2: Model Architecture
# Build a convolutional neural network (CNN) for image classification
model = tf.keras.Sequential([
    # Convolutional layer with 32 filters, each 3x3 pixels, using ReLU activation
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    # Max pooling layer with 2x2 window to reduce spatial dimensions
    tf.keras.layers.MaxPooling2D(2, 2),
    # Convolutional layer with 64 filters, each 3x3 pixels, using ReLU activation
    #tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # Max pooling layer with 2x2 window to reduce spatial dimensions
    #tf.keras.layers.MaxPooling2D(2, 2),
    # Convolutional layer with 128 filters, each 3x3 pixels, using ReLU activation
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    # Max pooling layer with 2x2 window to reduce spatial dimensions
    tf.keras.layers.MaxPooling2D(2, 2),
    # Convolutional layer with 128 filters, each 3x3 pixels, using ReLU activation
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    # Max pooling layer with 2x2 window to reduce spatial dimensions
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten layer to convert 3D feature maps into a 1D vector
    tf.keras.layers.Flatten(),
    # Fully connected layer with 512 units using ReLU activation
    tf.keras.layers.Dense(512, activation='relu'),
    # Output layer with 1 unit using sigmoid activation for binary classification
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Step 3: Training
# Compile the model with binary cross-entropy loss and RMSprop optimizer
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])

# Train the model using training and validation generators
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Step 4: Evaluation
# Initialize ImageDataGenerator for test data normalization
test_datagen = ImageDataGenerator(rescale=1./255)

# Generate test dataset from directory with specified parameters
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(150, 150),  # Resize images to 150x150 pixels
    batch_size=32,  # Batch size for testing
    class_mode='binary'  # Binary classification (normal or pneumonia)
)

# Evaluate the trained model on the test dataset
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test accuracy: {test_acc}')