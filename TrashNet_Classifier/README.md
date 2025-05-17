# 🗑️ TrashNet Image Classification using Transfer Learning

This project implements an image classification pipeline to automatically identify types of trash using the [TrashNet dataset](https://www.kaggle.com/datasets/feyzazkefe/trashnet). The goal is to classify waste images into one of six categories to aid in smart recycling and waste management systems.


## 🧠 Model Used

This project leverages **transfer learning** with state-of-the-art CNN architecture:

- **MobileNet**  

Model was trained on a resized dataset (224x224) and fine-tuned on the TrashNet dataset for multi-class classification.

## 🗃️ Dataset

The dataset consists of labeled images categorized into the following **6 classes**:

- `cardboard`
- `glass`
- `metal`
- `paper`
- `plastic`
- `trash`

The dataset is split using an 80/20 train-validation ratio via the `splitfolders` library.

## ⚙️ Features & Workflow

### 1. Data Preparation
- Dataset downloaded via Kaggle API
- Images normalized and resized to 224x224
- Visualized sample images from each class

### 2. Model Training
- Transfer learning with `MobileNet` 
- Used `Adam` optimizer with scheduled fine-tuning:
  - Base training for 50 epochs
  - Fine-tuning with reduced learning rate
- Used `class_weight` to handle class imbalance

### 3. Evaluation
- Accuracy and loss graphs
- Confusion matrix and classification report
- ROC AUC and F1-score for multi-class evaluation

### 4. Libraries Used
- `TensorFlow` / `Keras`
- `Matplotlib`, `Seaborn`
- `scikit-learn`
- `splitfolders`

## 📊 Results

The model achieved good classification performance across all classes, with EfficientNet generally outperforming MobileNet.

- 📈 Training Accuracy: ~95%+
- 📉 Validation Accuracy: ~90%+
- 🧾 Classification Report and Confusion Matrix included in the notebook

## 🚀 How to Run

1. Upload `kaggle.json` to access the dataset.
2. Run the notebook from top to bottom in Google Colab or Jupyter.
3. Ensure GPU acceleration is enabled for faster training.


