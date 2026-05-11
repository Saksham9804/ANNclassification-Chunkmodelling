import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('churn_model.h5')

# Load scaler and encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

# Streamlit title
st.title("Customer Churn Prediction")

# User inputs
gender = st.selectbox("Gender", label_encoder_gender.classes_)

geography = st.selectbox(
    "Geography",
    onehot_encoder_geo.categories_[0]
)

age = st.number_input("Age", 18, 92)

balance = st.number_input("Balance")

credit_score = st.number_input("Credit Score")

tenure = st.number_input("Tenure", 0, 10)

num_of_products = st.number_input(
    "Number of Products",
    1,
    4
)

estimated_salary = st.number_input("Estimated Salary")

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)

# Create dataframe
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
    'Geography': [geography]
})

# Encode Gender
input_data['Gender'] = label_encoder_gender.transform(
    input_data['Gender']
)

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform(
    input_data[['Geography']]
).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Remove original Geography column
input_data = input_data.drop('Geography', axis=1)

# Combine encoded geography columns
input_data_encoded = pd.concat(
    [input_data.reset_index(drop=True),
     geo_encoded_df.reset_index(drop=True)],
    axis=1
)

# Match exact training column order
input_data_encoded = input_data_encoded.reindex(
    columns=scaler.feature_names_in_,
    fill_value=0
)

# Scale input
input_data_scaled = scaler.transform(
    input_data_encoded
)

# Prediction
if st.button("Predict Churn"):

    prediction = model.predict(input_data_scaled)

    churn_probability = prediction[0][0]

    st.write(
        f"Churn Probability: {churn_probability:.2f}"
    )

    if churn_probability > 0.5:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is unlikely to churn.")