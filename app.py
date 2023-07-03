import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model('model/model.h5')

# Load Encoder
encoder_path = 'model/encoder.joblib'
encoder = joblib.load(encoder_path)

# Load Scaler
scaler_path = 'model/scaler.joblib'
scaler = joblib.load(scaler_path)

# Load the SHAP image
shap_image_path = 'images/shap_plot.png'


def preprocess_input(user_input, encoder, scaler):
    # Create a DataFrame with user input
    df = pd.DataFrame(user_input, index=[0])

    # Step 1: Create a new column called 'bmi_category' based on the 'bmi' column values
    bmi_categories = ['underweight', 'normal weight', 'overweight', 'obese']
    bmi_ranges = [0, 18.5, 25, 30, float('inf')]
    df['bmi_category'] = pd.cut(df['bmi'], bins=bmi_ranges, labels=bmi_categories, right=False)
    df['bmi_category'] = df['bmi_category'].astype('object')

    # Step 2: Encode the categorical features
    df_categorical = encoder.transform(df.select_dtypes(include='object'))
    encoded_cols = encoder.get_feature_names_out(df.select_dtypes(include='object').columns)
    df_encoded = pd.DataFrame(df_categorical, columns=encoded_cols, index=df.index)

    # Step 3: Scale the numerical features
    df_numerical = scaler.transform(df.select_dtypes(exclude='object'))
    scaled_cols = df.select_dtypes(exclude='object').columns
    df_scaled = pd.DataFrame(df_numerical, columns=scaled_cols, index=df.index)

    # Step 4: Concatenate the categorical and numerical dataframes along the columns axis
    df_preprocessed = pd.concat([df_encoded, df_scaled], axis=1)

    return user_input, df_preprocessed


@tf.function
def predict(df_preprocessed):
    # Make predictions using the pre-trained model
    predictions = tf.squeeze(model(df_preprocessed), axis=-1)

    return predictions


def main():
    # Set the app title
    st.title('Insurance Premium Prediction🛡️')

    # Create input fields for the user
    age = st.number_input('Age', min_value=0, max_value=120, value=10)
    sex = st.selectbox('Sex', ['male', 'female'])
    bmi = st.number_input('BMI', min_value=0.0, value=25.0)
    children = st.number_input('Children', min_value=0, max_value=10, value=0)
    smoker = st.selectbox('Smoker', ['no', 'yes'])
    region = st.selectbox('Region', ['southwest', 'southeast', 'northwest', 'northeast'])

    # Store the user input in a dictionary
    user_input = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }

    # Preprocess the user input
    user_input, input_df = preprocess_input(user_input, encoder, scaler)

    if st.button('Predict'):
        # Make predictions
        predictions = predict(input_df)

        # Display the predicted insurance cost
        st.write("Predicted Insurance Premium:", int(predictions), "$")


if __name__ == '__main__':
    main()
