# 🐱🐶 Cats vs Dogs Image Classifier
### MobileNetV2 + Focal Loss | TensorFlow / Keras

A binary image classifier that distinguishes cats from dogs using transfer learning on MobileNetV2, trained on the Microsoft Cats vs Dogs dataset (~25,000 images). The model is exported in both `.h5` and `.tflite` formats.

---

## 👥 Contributors

### 1. Rishi *(Project Lead)*
Rishi designed and implemented the core deep learning model. He chose MobileNetV2 as the base architecture and configured it with pretrained ImageNet weights. He wrote the custom **focal loss function** to handle class imbalance during training. He also structured the **two-phase training strategy** — first training only the top layers with the base frozen, then fine-tuning the deeper layers of MobileNetV2 at a lower learning rate (`1e-5`) while keeping BatchNormalization layers frozen to preserve learned statistics.

---

### 2. Dinesh *(Data Engineer)*
Dinesh handled everything related to getting the raw dataset ready for training. He wrote the code that checks for the ZIP file, extracts it, and scans through the `Cat` and `Dog` folders. He added logic to **filter out non-image files** like `Thumbs.db` and used PIL's `img.verify()` to **detect and delete corrupt images** that would crash training. He then used `train_test_split` to divide the valid images into an **80% training / 20% validation split** and organised them into the correct folder structure using `shutil.copy`.

---

### 3. Hari Om Yadav *(Training Pipeline & Evaluation)*
Hari Om set up the **data augmentation and loading pipeline** using Keras `ImageDataGenerator`. For training data he applied rescaling, random rotations (±20°), horizontal flips, and zoom to improve generalisation. For validation he applied only rescaling to ensure unbiased evaluation. He also wrote the **evaluation section** — combining the history from both training phases to plot accuracy and loss curves, generating the **confusion matrix**, and printing the full **classification report** with precision, recall, and F1-score for both classes.

---

### 4. Rohith *(Export & Deployment)*
Rohith handled saving and exporting the trained model for real-world use. He saved the final model in **Keras `.h5` format** (`mobilenet_final.h5`) for reuse in Python. He then used `TFLiteConverter` to convert and export the model as a **`.tflite` file** (`model.tflite`), making it ready for deployment on Android phones, Raspberry Pi, and other edge devices.

---

## 📁 Project Structure

```
cats_dogs_classifier.py   ← Main script
README.md                 ← This file

PetImages/                ← Extracted from ZIP (auto-created by Dinesh's code)
├── Cat/
└── Dog/

cats_dogs_split/          ← Train/val split (auto-created by Dinesh's code)
├── train/
│   ├── cat/
│   └── dog/
└── validation/
    ├── cat/
    └── dog/

mobilenet_final.h5        ← Keras model saved by Rohith's code
model.tflite              ← TFLite model saved by Rohith's code
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Install dependencies
```bash
pip install tensorflow scikit-learn matplotlib pillow
```

### Dataset
Download from Kaggle and place the ZIP in the project folder:

🔗 https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset

Expected filename: `kagglecatsanddogs_5340.zip`

The script handles extraction, cleaning, and splitting automatically on first run.

### Run
```bash
python cats_dogs_classifier.py
```

---

## 🧠 Model Architecture

```
Input (160×160×3)
    └── MobileNetV2 (pretrained ImageNet weights)
        └── GlobalAveragePooling2D
            └── Dense(128, ReLU)
                └── Dropout(0.2)
                    └── Dense(1, Sigmoid)  →  0 = Cat  |  1 = Dog
```

| Phase | Epochs | Layers Trained | Learning Rate |
|-------|--------|----------------|---------------|
| Phase 1 | 5 | Top layers only | Adam `1e-3` |
| Phase 2 | 5 | All except first 100 + BatchNorm | Adam `1e-5` |

---

## 📦 Output Files

| File | Description |
|------|-------------|
| `mobilenet_final.h5` | Full Keras model for Python inference |
| `model.tflite` | Lightweight model for mobile & edge devices |

---

## 📜 License

Educational use. Dataset © Microsoft, hosted on Kaggle under their respective terms.

---

*Built by Rishi, Dinesh, Hari Om Yadav & Rohith*
