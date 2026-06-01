import os
import zipfile
import shutil
import itertools
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =====================================================
# 1. PREPARE DATASET (Manual ZIP Check)
# =====================================================
zip_path = "kagglecatsanddogs_5340.zip"
extract_dir = "PetImages"
target_dir = "cats_dogs_split"

# Check if the ZIP exists before doing anything
if not os.path.exists(zip_path) and not os.path.exists(target_dir):
    print(f"ERROR: {zip_path} not found in the directory!")
    print("Please ensure the downloaded ZIP is in: " + os.getcwd())
    exit()

if not os.path.exists(target_dir):
    print("ZIP found! Starting extraction (this may take a minute)...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
    except Exception as e:
        print(f"Extraction failed: {e}")
        exit()
    
    print("Cleaning corrupt images and organizing...")
    for label in ["Cat", "Dog"]:
        folder = os.path.join(extract_dir, label)
        # Filter out non-jpg files like 'Thumbs.db'
        images = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]
        
        valid_images = []
        for f in images:
            f_path = os.path.join(folder, f)
            try:
                # Open and verify image integrity
                with Image.open(f_path) as img:
                    img.verify() 
                valid_images.append(f)
            except:
                os.remove(f_path) # Delete if corrupt

        # Split into Train (80%) and Validation (20%)
        train_imgs, val_imgs = train_test_split(valid_images, test_size=0.2, random_state=42)
        
        for split, split_imgs in [("train", train_imgs), ("validation", val_imgs)]:
            path = os.path.join(target_dir, split, label.lower())
            os.makedirs(path, exist_ok=True)
            for img in split_imgs:
                shutil.copy(os.path.join(folder, img), os.path.join(path, img))
    print("Dataset successfully organized.")
else:
    print("Dataset directory already exists. Skipping extraction.")

# =====================================================
# 2. DATA GENERATORS (Environment Setup)
# =====================================================
IMG_HEIGHT, IMG_WIDTH = 160, 160
BATCH_SIZE = 32

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, rotation_range=20, horizontal_flip=True, zoom_range=0.2
)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(target_dir, "train"),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(target_dir, "validation"),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False 
)

# =====================================================
# 3. DEFINE FOCAL LOSS & MODEL
# =====================================================
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        focal_weight = tf.pow(1 - p_t, gamma)
        alpha_t = (y_true * alpha) + ((1 - y_true) * (1 - alpha))
        return tf.reduce_mean(-alpha_t * focal_weight * tf.math.log(p_t))
    return focal_loss_fixed

base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                         include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])

# =====================================================
# 4. TRAINING (Phase 1 & Phase 2)
# =====================================================
print("Phase 1: Training top layers...")
history = model.fit(train_generator, epochs=5, validation_data=val_generator)

print("Phase 2: Fine-tuning base model...")
base_model.trainable = True
for layer in base_model.layers[:100]:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), 
              loss=focal_loss(), metrics=['accuracy'])

history_fine = model.fit(train_generator, epochs=5, validation_data=val_generator)

# =====================================================
# 5. EVALUATION & VISUALIZATION
# =====================================================
# Accuracy/Loss Curves
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

# Confusion Matrix
val_generator.reset()
y_pred_probs = model.predict(val_generator, steps=len(val_generator))
y_pred = (y_pred_probs > 0.5).astype(int)
y_true = val_generator.classes[:len(y_pred)]
cm = confusion_matrix(y_true, y_pred)

print("\n--- Final Evaluation ---")
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_true, y_pred))
plt.show()
layers.Dropout(0.2)

# =====================================================
# 6. EXPORT
# =====================================================
model.save('mobilenet_final.h5')
print("Saved Keras model to mobilenet_final.h5")

# TFLite Conversion
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
print("Saved TFLite model to model.tflite")