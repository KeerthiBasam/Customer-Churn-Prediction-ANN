import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import os

class ChurnModel(nn.Module):
    def __init__(self, input_size):
        super(ChurnModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

label_encoder_gender_path = os.path.join(BASE_DIR, 'label_encoder_gender.pkl')
onehot_encoder_geo_path = os.path.join(BASE_DIR, 'onehot_encoder_geo.pkl')
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
model_path = os.path.join(BASE_DIR, 'model.pth')

with open(label_encoder_gender_path, 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open(onehot_encoder_geo_path, 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

input_size = scaler.mean_.shape[0]
model = ChurnModel(input_size)

state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

if st.button('Predict'):
    gender_encoded = label_encoder_gender.transform([gender])[0]
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender_encoded],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    input_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)

    input_scaled = scaler.transform(input_data)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    with torch.no_grad():
        prediction = model(input_tensor)

    if prediction.item() > 0.5:
        st.write('Customer is likely to exit the bank')
    else:
        st.write('Customer is not likely to exit the bank')