import os
import shap
import matplotlib.pyplot as plt

# Defining a class called SHAPSummaryPlot
class SHAPSummaryPlot:
    
    # Initializing an instance of the class
    def __init__(self, ann_model, input_df):
        
        # Storing the ann_model object
        self.ann_model = ann_model
        
        # Storing the input_df object
        self.input_df = input_df
        
        # Initializing the shap_values attribute as None
        self.shap_values = None
    
    # Method to create the SHAP explainer and compute SHAP values
    def create_explainer(self):
        
        # Creating a DeepExplainer object using the ann_model's model and the input_df values
        explainer = shap.DeepExplainer(self.ann_model.model, self.input_df.values)
        
        # Computing the SHAP values for the input_df values
        self.shap_values = explainer.shap_values(self.input_df.values)
    
    # Method to save the SHAP summary plot
    def save_plot(self):
        
        # Defining the directory path for the images
        image_dir = os.path.join("..", "images")
        
        # Defining the file path for the plot
        file_path = os.path.join(image_dir, "shap_plot.png")
        
        # Creating the directory for the images if it doesn't exist
        os.makedirs(image_dir, exist_ok=True)
        
        # Generating the SHAP summary plot with the computed SHAP values and input_df, using bar plot type and not showing the plot
        shap.summary_plot(self.shap_values, self.input_df, plot_type="bar", show=False)
        
        # Setting the x-label of the plot
        plt.gca().set_xlabel('SHAP Value')
        
        # Removing the legend from the plot
        plt.gca().legend().remove()
        
        # Adjusting the plot layout for better visualization
        plt.tight_layout()
        
        # Saving the plot to the specified file path
        plt.savefig(file_path)
        
        # Closing the plot
        plt.close()
        
        # Printing a success message
        print("Plot saved successfully.")
