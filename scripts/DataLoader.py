import pandas as pd
import os

# Defining a class called InsuranceData
class InsuranceData:
    
    # Initializing an instance of the class
    def __init__(self):
        
        # Constructing the file path to the insurance.csv file using os.path.join
        file_path = os.path.join("..", "data", "insurance.csv")
        
        # Reading the CSV file using pandas' read_csv function and storing it in the df attribute
        self.df = pd.read_csv(file_path)
    
    # Method to get the data stored in the df attribute
    def get_data(self):
        
        # Printing a success message
        print("Data Loaded successfully.")
        
        return self.df