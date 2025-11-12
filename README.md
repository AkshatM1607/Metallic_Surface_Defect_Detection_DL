# 🧠 Metal Surface Defect Detection using Machine Learning and Deep Learning

This project presents a comprehensive **AI-driven quality control system** for detecting **metal surface defects** using **Machine Learning (ML)** and **Deep Learning (DL)** techniques.  
The study compares multiple models — **Convolutional Neural Network (CNN)**, **Artificial Neural Network (ANN)**, **Support Vector Machine (SVM)**, and **Genetic Algorithm optimized SVM (SVM+GA)** — to classify six common defect types found in industrial metal surfaces.

---

## 🏭 Problem Statement

In manufacturing industries (automotive, aerospace, construction), **manual surface inspection** is slow, inconsistent, and prone to human error.  
This project automates **metal surface defect detection** using **computer vision** and **AI models**, enabling:
- Faster real-time defect analysis  
- Reduced inspection cost  
- Improved product reliability  

---

## ⚙️ Defect Classes

The dataset includes **grayscale images (200×200 px)** with the following defect types:

| Class No. | Defect Type | Description |
|------------|--------------|-------------|
| 1 | Crazing | Fine crack-like lines |
| 2 | Inclusion | Embedded foreign material |
| 3 | Patches | Irregular surface textures |
| 4 | Pitted | Small holes/depressions |
| 5 | Rolled | Linear rolling marks |
| 6 | Scratches | Abrasion or contact marks |

---

## 📊 Dataset and Preprocessing

- **Image Size:** 200×200 (grayscale)  
- **Normalization:** Pixel values scaled to [0,1]  
- **Augmentation:** Random rotation, shift, and horizontal flip  
- **Split:** 80% training, 20% testing  
- **Batch Size:** 52  

For SVM, features were extracted using **VGG16 pretrained model (ImageNet weights)** to improve performance.

---

## 🧮 Models Implemented

### 🧩 1. Convolutional Neural Network (CNN)
- 3 Conv layers (32, 64, 128 filters, 3×3 kernel)
- MaxPooling (2×2), BatchNorm, Dropout(0.5)
- Dense layers: 512 → 256 → 128 → 6 (Softmax)
- **Optimizer:** Adam (lr=0.001)
- **Epochs:** 25  
- ✅ **Accuracy:** **95.83%**

---

### 🔢 2. Artificial Neural Network (ANN)
- Flattened input (40,000 features)
- Dense layers: 512 → 256 → 128
- Dropout(0.5)
- **Accuracy:** 88.89%
- Fast inference and suitable for real-time systems.

---

### ⚙️ 3. Support Vector Machine (SVM)
- Kernel: RBF  
- Parameters: C = 0.5, γ = 0.005  
- Used **VGG16 feature extraction**
- **Accuracy:** 82%

---

### 🧬 4. Genetic Algorithm Optimized SVM (SVM + GA)
- Optimized hyperparameters via GA:
  - **C = 9.0883**, **γ = 0.0053**
- Population: 50, Generations: 20  
- **Accuracy:** 87.50%
- GA improved SVM performance but increased training time (187 mins).

---

## 📈 Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|-----------|---------|-----------|
| **CNN** | **95.83%** | **0.96** | **0.96** | **0.96** |
| ANN | 88.89% | 0.89 | 0.89 | 0.89 |
| SVM | 82.00% | 0.83 | 0.82 | 0.82 |
| SVM + GA | 87.50% | 0.88 | 0.88 | 0.88 |

**Training & Inference Time:**

| Model | Training (min) | Inference (ms/image) |
|--------|----------------|----------------------|
| CNN | 58.5 | 8.7 |
| ANN | 23.2 | 5.3 |
| SVM | 12.7 | 14.2 |
| GA | 187.6 | 14.2 |

---

## 🧠 Key Takeaways

- **CNN** achieves the highest accuracy and generalization.  
- **SVM+GA** proves the importance of hyperparameter optimization.  
- **ANN** offers a balance between performance and speed.  
- Demonstrates how **Deep Learning can outperform traditional ML** in industrial defect detection.  

---

## 🧰 Tech Stack

- **Language:** Python 3.12  
- **Frameworks:** TensorFlow, Keras, Scikit-learn  
- **Libraries:** NumPy, Pandas, Matplotlib, OpenCV  
- **Hardware:** AMD Ryzen 5600H, 16GB RAM, Radeon GPU  
- **OS:** Windows 11  

---

## 🧪 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/AkshatM1607/Metallic_Surface_Defect_Detection_DL.git
   cd Metallic_Surface_Defect_Detection_DL
