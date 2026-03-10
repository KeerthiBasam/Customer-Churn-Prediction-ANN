# Customer Churn Prediction using Artificial Neural Networks (ANN)

## Project Overview

Customer churn prediction is an important problem for businesses, especially in industries such as banking, telecommunications, and subscription services. Retaining existing customers is often more cost-effective than acquiring new ones.

This project builds a **machine learning system using an Artificial Neural Network (ANN)** to predict whether a bank customer is likely to leave the bank (churn). The model is trained on customer demographic and financial data and deployed through an **interactive Streamlit web application** where users can input customer information and receive real-time churn predictions.

The objective of this project is to demonstrate an **end-to-end machine learning pipeline**, including data preprocessing, model training, model saving, and deployment.

---

## Problem Statement

Banks often struggle to identify customers who may leave their services. By analyzing historical customer data, we can train a predictive model that estimates the probability of churn.

This project predicts **whether a customer will churn or stay**, based on several features such as credit score, geography, age, balance, and account activity.

---

## Dataset

The dataset used in this project contains customer information from a bank. Each row represents a customer and includes several attributes that may influence churn behavior.

### Key Features

* **CreditScore** – Customer's credit score
* **Geography** – Customer’s country
* **Gender** – Male or Female
* **Age** – Age of the customer
* **Tenure** – Number of years the customer has stayed with the bank
* **Balance** – Account balance
* **NumOfProducts** – Number of bank products used by the customer
* **HasCrCard** – Whether the customer has a credit card
* **IsActiveMember** – Whether the customer is an active member
* **EstimatedSalary** – Estimated annual salary

### Target Variable

* **Exited**

  * 1 → Customer churned
  * 0 → Customer stayed

---

## Machine Learning Pipeline

The complete workflow for this project is shown below:

Data Collection
↓
Data Cleaning and Preprocessing
↓
Feature Encoding
↓
Feature Scaling
↓
Train Artificial Neural Network
↓
Model Evaluation
↓
Save Model and Preprocessing Objects
↓
Deploy with Streamlit
↓
Real-time Predictions

---

## Data Preprocessing

Before training the neural network, several preprocessing steps were performed:

### 1. Label Encoding

The **Gender** feature is converted into numerical values using `LabelEncoder`.

Example:

Male → 1
Female → 0

### 2. One-Hot Encoding

The **Geography** feature is transformed into multiple binary columns using `OneHotEncoder`.

Example:

France → [1,0,0]
Germany → [0,1,0]
Spain → [0,0,1]

### 3. Feature Scaling

Numerical features are scaled using **StandardScaler** so that all features have similar ranges. This improves neural network performance.

---

## Model Architecture

The churn prediction model is implemented using **TensorFlow/Keras** with an Artificial Neural Network.

### ANN Structure

Input Layer
↓
Hidden Layer (ReLU Activation)
↓
Hidden Layer (ReLU Activation)
↓
Output Layer (Sigmoid Activation)

The **sigmoid activation function** in the output layer produces a probability value between **0 and 1** indicating the likelihood of customer churn.

Example:

0.15 → Customer likely to stay
0.82 → Customer likely to churn

---

## Technologies Used

* Python
* TensorFlow / Keras
* Scikit-learn
* Pandas
* NumPy
* Streamlit
* Pickle

These tools were used for data processing, model training, and deployment.

---

## Project Structure

```
customer-churn-prediction-ann/

data/
    Churn_Modelling.csv

notebooks/
    experiments.ipynb
    hyperparameter_tuning_ann.ipynb
    prediction.ipynb

models/
    model.h5

encoders/
    label_encoder_gender.pkl
    onehot_encoder_geo.pkl
    scaler.pkl

app.py
requirements.txt
README.md
```

---

## Web Application

A **Streamlit web application** is used to interact with the trained model.

Users can input customer details such as:

* Geography
* Gender
* Age
* Credit Score
* Balance
* Estimated Salary
* Tenure
* Number of Products
* Credit Card ownership
* Active membership status

After entering these values, the model predicts the **probability of customer churn** and displays whether the customer is likely to leave the bank.

---

## Running the Project

### Step 1: Clone the Repository

```
git clone https://github.com/yourusername/customer-churn-prediction-ann.git
cd customer-churn-prediction-ann
```

### Step 2: Install Dependencies

```
pip install -r requirements.txt
```

### Step 3: Run the Streamlit Application

```
streamlit run app.py
```

This will start the application locally, and it will open in your browser.

---

## Example Prediction

Example Customer Input:

Credit Score: 650
Age: 40
Balance: 60000
Number of Products: 2
Active Member: Yes

Model Output:

Churn Probability: 0.67

Prediction: **Customer likely to churn**

---

## Future Improvements

Possible improvements for this project include:

* Hyperparameter tuning for improved model performance
* Adding model evaluation metrics such as ROC-AUC and confusion matrix
* Deploying the application using Docker
* Hosting the application on cloud platforms such as AWS or GCP
* Implementing model monitoring and logging

---

## Author

This project was developed as part of a **machine learning portfolio project** to demonstrate skills in:

* Data preprocessing
* Artificial Neural Networks
* Machine learning pipelines
* Model deployment using Streamlit
