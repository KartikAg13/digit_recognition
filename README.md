# Digit Recognition using CNN

This repository demonstrates how to build a Convolutional Neural Network (CNN) using TensorFlow to perform digit recognition on the MNIST dataset. The notebook walks through all the necessary steps, from importing libraries to saving the trained model. The filename for the main code is `main.ipynb`.

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Steps in the Notebook](#steps-in-the-notebook)
4. [Results](#results)
5. [Usage](#usage)
6. [Saved Model](#saved-model)

---

## **Introduction**
This project utilizes the MNIST dataset to recognize handwritten digits (0-9) using a CNN. The notebook explains step-by-step how the network is built, trained, and evaluated.

---

## **Dataset**
The dataset used is the **MNIST (Modified National Institute of Standards and Technology)** dataset:
- **Training Data**: 60,000 grayscale images (28x28 pixels).
- **Testing Data**: 10,000 grayscale images (28x28 pixels).

Each image represents a handwritten digit, and the goal is to classify each image into one of 10 classes (0-9).

---

## **Steps in the Notebook**

### **1. Import Libraries**
Essential libraries such as TensorFlow, NumPy, and Matplotlib are imported for building and visualizing the model.

### **2. Load the Dataset**
The MNIST dataset is loaded and its shapes are printed to verify the dimensions of the data.

### **3. Visualize the Dataset**
Several sample images from the dataset are displayed using Matplotlib for better understanding.

### **4. Preprocess the Dataset**
- The pixel values of the images are normalized to the range [0, 1].
- The dataset is reshaped to include a channel dimension (required by CNNs).

### **5. Create the Neural Network**
A CNN architecture is defined using TensorFlow:
- **2 Convolutional Layers** with MaxPooling.
- **1 Dense Hidden Layer**.
- **1 Output Layer** using Softmax activation.

### **6. Compile the Model**
The model is compiled with:
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metric**: Accuracy

### **7. Visualize the Model**
A graphical representation of the model is generated using the `plot_model` function.

### **8. Train the Network**
The model is trained on the training dataset for 10 epochs with validation data.

### **9. Evaluate the Model**
The trained model's accuracy is tested on the test dataset.

### **10. Visualize Training Results**
- Plots of training/validation accuracy and loss over epochs are generated to visualize the model's learning process.

### **11. Save the Model**
The trained model is saved in HDF5 format (`mnist_cnn_model.h5`) for future use.

---

## **Results**
- The model achieved a **test accuracy of approximately 99%**, demonstrating excellent performance on the MNIST dataset.
- Plots of accuracy and loss confirm effective training with minimal overfitting.

---

## **Usage**
1. Clone the repository:
   ```bash
   git clone <repo-link>
   cd <repo-folder>
   ```

2. Open the Jupyter Notebook `main.ipynb`:
   ```bash
   jupyter notebook main.ipynb
   ```

3. Follow the step-by-step instructions in the notebook to:
   - Train the model.
   - Evaluate its performance.
   - Save the trained model.

---

## **Saved Model**
The trained model is saved as `mnist_cnn_model.h5` in the repository. You can load it in another program to make predictions on new data:

```python
from tensorflow.keras.models import load_model
model = load_model('mnist_cnn_model.h5')
predictions = model.predict(new_data)
```

---

This concludes the steps for recognizing digits using a CNN. Happy learning!