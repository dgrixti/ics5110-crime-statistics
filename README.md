# US Crime Analysis: Weapon Use Prediction

Welcome to the **Weapon Use Prediction** project! This interactive web application leverages machine learning models to predict the type of weapon (Firearm vs. Non-Firearm) used in crimes based on demographic and contextual features. The app is built using **Gradio** and hosted on **Hugging Face Spaces**.

Latest code: https://huggingface.co/spaces/grixtid/ICS5110/tree/main

---

## 🔍 **Overview**

The primary goal of this project is to analyze crime data and classify weapon usage using machine learning techniques. The application supports multiple models, including:

- **Random Forest Classifier**
- **XGBoost**
- **Logistic Regression**
- **Artificial Neural Network (ANN)**

The models are trained on a publicly available crime dataset, which includes features such as **Region**, **Victim Age**, **Relationship Type**, and more. The dataset is imbalanced - 102,988 records (67.09%) for "Firearm" compared to approximately 50,523 (32.91%) for "Non-Firearm", and techniques like **SMOTE** were applied to improve model performance.

---

## 🚀 **Features**

- **Interactive Interface**: Enter key features like Region, Season, Relationship Type, etc., and get predictions in real time.
- **Multiple Models**: Compare predictions from different machine learning algorithms.
- **Performance Metrics**: View Accuracy, Precision, Recall, F1-Score, and Confusion Matrix for model performance evaluation.
- **Real-World Use Case**: Designed for researchers and law enforcement agencies to analyze crime patterns.

---

## 💻 **How to Use**

1. **Access the Application**:
   - Visit the [Hugging Face Space](#) to interact with the app. The iframe below also provides direct access.

2. **Input Features**:
   - Select or input values for features like Region, Victim Sex, Relationship Type, etc.
   - Adjust Victim Age using the slider.

3. **Choose a Model**:
   - Select a model from the dropdown menu: Random Forest, XGBoost, Logistic Regression, or ANN.

4. **View Predictions**:
   - The app displays the predicted weapon category (Firearm or Non-Firearm) along with detailed metrics.

---

## 📊 **Technical Details**

### **Dataset**
- **Source**: [US Crime Dataset on Kaggle](https://www.kaggle.com/datasets/mrayushagrawal/us-crime-dataset)
- **Features**: Demographics, geographic data, relationships, and crime specifics.

### **Models**
- **Random Forest**: Robust ensemble method with SMOTE for handling imbalance.
- **XGBoost**: Boosting algorithm with optimized hyperparameters for precision-recall balance.
- **Logistic Regression**: Lightweight model with decent accuracy.
- **ANN**: Neural network with three dense layers for binary classification.

### **Preprocessing**
- Categorical features: Encoded using LabelEncoder.
- Numerical features: Standardized using StandardScaler.
- Class imbalance: Addressed using SMOTE for minority-class oversampling.

---

## 🌐 **Try It Out**

### **Hugging Face Space**

<iframe
	src="https://grixtid-ics5110.hf.space"
    width="100%"
    height="600"
    frameborder="0"
    allowfullscreen>
</iframe>