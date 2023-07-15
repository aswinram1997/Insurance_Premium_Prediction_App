# Insurance_Premium_Prediction_App

[My Streamlit app can be found here!](<https://insurancepremiumpredictionapp.streamlit.app>) 

![pexels-mikhail-nilov-7731330](https://github.com/aswinram1997/DataScience_Portfolio/assets/102771069/b21d0440-5ca8-4ebc-9116-026cfe01b29c)

## Project Overview
The health insurance industry faces the challenge of premium prediction, and leveraging ML is crucial in addressing this issue. While the [Kaggle Dataset](<https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset>) used in this project may be common, the true innovation lies in developing a [Streamlit web app](<https://insurancepremiumpredictionapp.streamlit.app>) using object-oriented programming principles. This approach ensures easy maintenance of the code, updates to the database, and continuous improvement of the model with additional data to meet the evolving needs of customers. The ML workflow compares linear regression and artificial neural networks (ANN), with the selected model integrated into the web app. This app enables fair pricing, customer satisfaction, and a competitive edge, while the object-oriented programming principles ensure a well-structured, modular, and adaptable codebase for seamless updates and data-driven decision making. Ultimately, the app provides insurance companies with a powerful tool to accurately predict premiums, streamline processes, and meet the changing demands of the insurance landscape.

## Dataset Overview
The [Kaggle Dataset](<https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset>) consists of 1338 rows of insured data, including attributes such as Age, Sex, BMI, Number of Children, Smoker, and Region. The insurance charges associated with each insured individual are also provided. The dataset is free from missing or undefined values, ensuring data integrity for analysis.

## Methodology
The project follows a specific workflow for insurance premium prediction using the provided dataset:

- Data Collection:The dataset containing insured data, including attributes and insurance charges, is collected as the initial step.

- Exploratory Data Analysis (EDA):EDA is conducted to gain insights into the dataset, identify patterns, and understand the relationships between attributes and insurance charges. This analysis provides valuable information for feature selection and modeling.

- Data Splitting:The preprocessed dataset is split into training, validation, and testing sets. The training set and validation set is used to train the prediction models, the testing set is used for evaluation.

- Data Preprocessing:Data preprocessing involves several steps, including feature engineering, encoding categorical variables, and scaling numerical features. Feature engineering may involve creating new features or transforming existing ones to better represent the underlying relationships.

- Modeling, Evaluation, and Interpretation:Two prediction models, linear regression, and an ANN, are trained, evaluated, and interpreted with SHAP values using the training, validation, and testing data. The performance of each model is assessed using the R2 score. The ANN is identified as the winning algorithm based on superior performance.

## Results
The results indicate that both the linear regression and ANN models demonstrated similar generalization capabilities. However, the ANN model exhibited better overall performance in accurately predicting insurance premiums, as reflected by higher R2 scores across the train, validation, and test sets. This suggests that the ANN model is not only accurate but also effectively captures the underlying patterns and relationships in the data, making it a preferred choice for insurance premium estimation.

## Conclusion
The development of the Streamlit web app utilizing object-oriented programming for insurance premium prediction follows a specific methodology. This includes data collection, exploratory data analysis, data preprocessing, data splitting, modeling, and evaluation. By employing object-oriented programming, the project enhances code organization, reusability, and maintainability. The ANN model emerges as the preferred algorithm for insurance premium prediction based on its superior performance. The web app provides insurance companies with a valuable tool for accurate premium estimation, enabling optimal pricing strategies, risk management, and data-driven decision-making. 



