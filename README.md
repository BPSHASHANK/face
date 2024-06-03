# face
face recognition model using TensorFlow 
This code sets a local path for the dataset, manually extracts it using the `tarfile` module, and loads it with `ImageDataGenerator`. The model is built using MobileNetV2 with custom layers for face recognition. The model is trained, and the training history is plotted. Finally, the model is evaluated on validation data and saved. This approach ensures that the dataset is correctly extracted and utilized for training the face recognition model.
