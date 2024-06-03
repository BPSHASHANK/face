# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

# Ensure TensorFlow version is compatible
print("TensorFlow version:", tf.__version__)

# Download and prepare the dataset
# For this example, let's use the LFW (Labeled Faces in the Wild) dataset
# Alternatively, you can use your own dataset

data_url = "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
data_dir = tf.keras.utils.get_file(origin=data_url, fname="lfw", untar=True)

# Create a data generator for loading the data
data_gen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

train_data = data_gen.flow_from_directory(data_dir, target_size=(224, 224),
                                          batch_size=32, subset='training')

val_data = data_gen.flow_from_directory(data_dir, target_size=(224, 224),
                                        batch_size=32, subset='validation')

# Load the pretrained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_data.num_classes, activation='softmax')(x)

# Construct the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model architecture
model.summary()

# Train the model
history = model.fit(train_data, epochs=10, validation_data=val_data)

# Plot the training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Evaluate the model on the validation data
val_loss, val_accuracy = model.evaluate(val_data)
print(f'Validation loss: {val_loss}')
print(f'Validation accuracy: {val_accuracy}')

# Save the model
model.save('face_recognition_model.h5')
