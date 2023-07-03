import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
import joblib

# Defining a class called DataPreprocessor
class DataPreprocessor:
    
    # Initializing an instance of the class
    def __init__(self):
        
        # Defining a list of BMI categories
        self.bmi_categories = ['underweight', 'normal weight', 'overweight', 'obese']
        
        # Defining the ranges for each BMI category
        self.bmi_ranges = [0, 18.5, 25, 30, float('inf')]
        
        # Initializing an instance of the OneHotEncoder class
        self.encoder = OneHotEncoder(sparse=False)
        
        # Initializing an instance of the StandardScaler class
        self.scaler = StandardScaler()
        
        # Initializing the df attribute as None
        self.df = None
        
        # Initializing the y attribute as None
        self.y = None
        
                
    # Method to create a BMI category column in the dataframe
    def create_bmi_category(self, df):
        
        # Creating a new column called 'bmi_category' based on the 'bmi' column values
        df['bmi_category'] = pd.cut(df['bmi'], bins=self.bmi_ranges, labels=self.bmi_categories, right=False)
        
        # Changing the data type of the 'bmi_category' column to object
        df['bmi_category'] = df['bmi_category'].astype('object')
        
    # Method to split the inputs and output from the dataframe
    def split_inputs_and_output(self, df):
        
        # Storing the 'charges' column values in the y attribute as a numpy array
        self.y = np.array(df['charges'])
        
        # Removing the 'charges' column from the dataframe
        df.drop(['charges'], axis=1, inplace=True)
        
    # Method to encode categorical features in the dataframe
    def encode_categorical_features(self, df):
        
        # Encoding the categorical features using the OneHotEncoder and storing the result in df_encoded
        df_encoded = self.encoder.fit_transform(df.select_dtypes(include='object'))
        
        # Getting the feature names for the encoded columns
        encoded_cols = self.encoder.get_feature_names_out(df.select_dtypes(include='object').columns)
        
        # Creating a new dataframe with the encoded columns and using the same index as the original dataframe
        df_encoded = pd.DataFrame(df_encoded, columns=encoded_cols, index=df.index)
        
        # Returning the encoded dataframe
        return df_encoded
        
    # Method to scale numerical features in the dataframe
    def scale_numerical_features(self, df):
        
        # Scaling the numerical features using the StandardScaler and storing the result in df_scaled
        df_scaled = self.scaler.fit_transform(df.select_dtypes(exclude='object'))
        
        # Getting the column names for the scaled columns
        scaled_cols = df.select_dtypes(exclude='object').columns
        
        # Creating a new dataframe with the scaled columns and using the same index as the original dataframe
        df_scaled = pd.DataFrame(df_scaled, columns=scaled_cols, index=df.index)
        
        # Returning the scaled dataframe
        return df_scaled
        
    # Method to concatenate the categorical and numerical features
    def concatenate_features(self, df_categorical, df_numerical):
        
        # Concatenating the categorical and numerical dataframes along the columns axis
        df_concatenated = pd.concat([df_categorical, df_numerical], axis=1)
        
        # Storing the concatenated dataframe in the df attribute
        self.df = df_concatenated
        
    # Method to preprocess the dataframe
    def preprocess(self, df):
        
        # Creating the BMI category column
        self.create_bmi_category(df)
        
        # Splitting the inputs and output from the dataframe
        self.split_inputs_and_output(df)
        
        # Encoding the categorical features
        df_categorical = self.encode_categorical_features(df)
        
        # Scaling the numerical features
        df_numerical = self.scale_numerical_features(df)
        
        # Concatenating the categorical and numerical features
        self.concatenate_features(df_categorical, df_numerical)
        
        # Saving the encoder and scaler
        model_folder = "../model"  # Path to the model folder outside the existing folder
        os.makedirs(model_folder, exist_ok=True)
        
        encoder_path = os.path.join(model_folder, "encoder.joblib")
        scaler_path = os.path.join(model_folder, "scaler.joblib")
        
        joblib.dump(self.encoder, encoder_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Printing a success message
        print("Data Preprocessed successfully.")
        
        
        # Returning the preprocessed dataframe and the output values
        return self.df, self.y