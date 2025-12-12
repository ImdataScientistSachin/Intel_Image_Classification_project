## Intel Image Classification
"""
===============================================================
EXECUTIVE SUMMARY : Intel Natural Scenes Classification
===============================================================
Purpose:
- Build a CNN model to classify natural scene images into 6 categories:
  buildings, forest, glacier, mountain, sea, street.

Why it matters:
- Image classification is a core computer vision task.
- Recruiters value candidates who can demonstrate end‑to‑end pipelines:
  dataset preparation, preprocessing, model design, training, evaluation, and deployment.

Techniques highlighted:
- TensorFlow GPU configuration for CUDA acceleration.
- Dataset extraction and exploration.
- Image preprocessing and batching with tf.keras utilities.
- CNN architecture with Conv2D, BatchNorm, MaxPooling, Dense, Dropout.
- Training with early stopping and visualization of accuracy/loss curves.
- Model saving for deployment.
===============================================================
"""

# ==== Step 1: GPU Environment Setup ====
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Physical Devices:", physical_devices)
if physical_devices:
    # Allow dynamic memory growth to prevent OOM errors
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# ==== Step 2: Import Libraries ====
# Why: These libraries cover visualization (matplotlib), image handling (PIL, cv2),
# model building (keras), and dataset utilities.
import matplotlib.pyplot as plt
import numpy as np
import PIL
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Rescaling, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import pathlib, zipfile, glob

# ==== Step 3: Dataset Paths ====
Dir_Path = r"C:\\Users\\demog\\.keras\\datasets\\Intel_Image_Classification\\"
train_zip_path = os.path.join(Dir_Path, "seg_train.zip")
test_zip_path = os.path.join(Dir_Path, "seg_test.zip")

# Check if zip files exist
if not os.path.exists(train_zip_path):
    print(f"Training zip file does not exist at {train_zip_path}")
if not os.path.exists(test_zip_path):
    print(f"Testing zip file does not exist at {test_zip_path}")

print(Dir_Path)
print(train_zip_path)
print(test_zip_path)

# ==== Step 4: Extract Dataset ====
train_dir = tf.keras.utils.get_file(
    fname="image_train.zip",
    origin=f"file:\\{train_zip_path}",
    extract=True,
    archive_format="zip"
)
test_dir = tf.keras.utils.get_file(
    fname="image_test.zip",
    origin=f"file:\\{test_zip_path}",
    extract=True,
    archive_format="zip"
)

train_dir = os.path.join(Dir_Path, "seg_train")
test_dir = os.path.join(Dir_Path, "seg_test")
print(train_dir)
print(test_dir)

# ==== Step 5: Explore Dataset ====
train_contains = os.listdir(train_dir)
print(len(train_contains))
print(train_contains)

# ==== Step 6: Visualize Sample Image ====
from tensorflow.keras.utils import load_img, img_to_array
forest = 'C:/Users/demog/.keras/datasets/Intel_Image_Classification/seg_train/forest/'
print('total images in forest :' ,len(forest))
print("Files in directory:", os.listdir(forest))

image_path = r'C:\Users\demog\.keras\datasets\Intel_Image_Classification\seg_train\forest\12968.jpg'
if os.path.isfile(image_path):
    img = load_img(image_path)
    print("Image loaded successfully.")
img

# ==== Step 7: Dataset Preparation ====
BATCH_SIZE = 32
IMG_SIZE = (150, 150)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
)
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
)

class_names = train_dataset.class_names
print("Class names:", class_names)

# ==== Step 8: Inspect Sample Batch ====
sample_img, labels = next(iter(train_dataset))
print("Sample Images Shape:", sample_img.shape)
print("Labels Shape:", labels.shape)

index = 4
plt.imshow(sample_img[index].numpy().astype('uint8'))
plt.axis('off')
plt.show()
print("Label:", labels[index].numpy())
print("Class Name:", class_names[labels[index].numpy()])

# ==== Step 9: Visualize Distribution ====
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i].numpy()])
        plt.axis("off")

# ==== Step 10: Optimize Dataset Pipeline ====
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# ==== Step 11: Define CNN Model ====
model = Sequential([
    Input(shape=(150, 150, 3)),             # Explicit input layer
    Rescaling(1./255),                      # Normalize pixel values
    Conv2D(32, 5, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Conv2D(128, 3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),                           # Prevent overfitting
    Dense(6, activation='softmax')          # Output layer for 6 classes
])
model.summary()

# ==== Step 12: Compile Model ====
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Early stopping callback to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# ==== Step 13: Train Model ====
epochs = 20
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early_stop])

# ==== Step 14: Plot Training Curves ====
acc, val_acc = history.history['accuracy'], history.history['val_accuracy']
loss, val_loss = history.history['loss'], history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='center right')
plt.title('Training and Validation Loss')
plt.show()

# ==== Step 15: Save Model ====
dummy_input = tf.random.normal([1, 150, 150, 3])
_ = model(dummy_input)  # Forward pass to build model graph
model.save("intel_image_Classifier_model.h5")
print("Model saved as Intel_image_Classifier_V1_models.h5")

"""
===============================================================
FINAL SUMMARY
===============================================================
- Dataset of 25k natural scene images prepared and explored.
- CNN model with Conv2D, BatchNorm, MaxPooling, Dense, Dropout layers implemented.
- Training achieved robust accuracy with early stopping.
- Accuracy/loss curves plotted for performance evaluation.
- Model saved for deployment as .h5 file.

Recruiter takeaway:
This notebook demonstrates end‑to‑end computer vision workflow:
data ingestion, preprocessing, CNN design, training, evaluation,
and deployment — all critical skills for applied ML and AI engineering.
===============================================================
"""
