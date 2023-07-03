from DataLoader import InsuranceData
from DataPreprocessor import DataPreprocessor
from ModelTrainer import NeuralNetwork
from ShapAnalyzer import SHAPSummaryPlot
import warnings

# ignore warnings
warnings.filterwarnings("ignore")

# Step 1: Loading and getting the insurance data
data_loader = InsuranceData()  # Creating an instance of the InsuranceData class
data = data_loader.get_data()  # Getting the data using the get_data method

# Step 2: Preprocessing the data
preprocessor = DataPreprocessor()  # Creating an instance of the DataPreprocessor class
preprocessed_data, output_variable = preprocessor.preprocess(data)  # Preprocessing the data

# Step 3: Training and saving the neural network model
model_trainer = NeuralNetwork(preprocessed_data, output_variable)  # Creating an instance of the NeuralNetwork class and training the model
model_trainer.save_model()  # Saving the trained model

# Step 4: Analyzing and saving the SHAP summary plot
shap_analyzer = SHAPSummaryPlot(model_trainer, preprocessed_data)  # Creating an instance of the SHAPSummaryPlot class
shap_analyzer.create_explainer()  # Creating the SHAP explainer and computing SHAP values
shap_analyzer.save_plot()  # Saving the SHAP summary plot
