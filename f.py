import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import tarfile
#user can use url also here am used ma local file
# Define the path to the local file
local_data_path = r'C:\Users\shashank\Desktop\face\lfw-deepfunneled.tgz'

# Ensure the file exists
if not os.path.exists(local_data_path):
    raise FileNotFoundError(f"The file at path {local_data_path} does not exist.")

# Define the directory to extract the dataset
extract_dir = r'C:\Users\shashank\Desktop\face\lfw'

# Extract the dataset
if not os.path.exists(extract_dir):
    with tarfile.open(local_data_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)
    print(f"Dataset extracted to {extract_dir}")
else:
    print(f"Dataset already extracted to {extract_dir}")

# Create a data generator for loading the data
data_gen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

train_data = data_gen.flow_from_directory(extract_dir, target_size=(224, 224),
                                          batch_size=32, subset='training')

val_data = data_gen.flow_from_directory(extract_dir, target_size=(224, 224),
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
