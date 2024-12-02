** Housing Price Prediction using Linear Regression

**overview

This project implements a housing price prediction model using Linear Regression with TensorFlow. The model is trained on the Kaggle House Prices dataset and provides a simple graphical user interface (GUI) built with Streamlit, allowing users to input house details and receive predicted prices. This project is intended for educational purposes, demonstrating data preprocessing, model building, evaluation, and GUI development.

** Features

- Load and preprocess the Kaggle House Prices dataset.
- Handle missing values using mean and mode imputation.
- Normalize numerical features and encode categorical variables using target encoding.
- Train a Linear Regression model using TensorFlow.
- Evaluate model performance using Root Mean Square Error (RMSE).
- Visualize actual vs. predicted housing prices and residuals distribution.
- Interactive GUI for predicting housing prices based on user input.

** Requirements

- Python 3.6 or higher
- Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `tensorflow`
  - `streamlit`
  - `matplotlib`
  - `seaborn`

** Installation

1. Clone the Repository (if applicable):
 
   git clone <repository-url>
   cd <repository-directory>
2. Install Required Libraries: Use pip to install the necessary libraries:
pip install pandas numpy scikit-learn tensorflow streamlit matplotlib seaborn

3.Download the Dataset: Download the Kaggle House Prices dataset from Kaggle.

Save the train.csv file as house_prices.csv in the same directory as the Python script.

Usage
on your terminal:
streamlit run housing_price_prediction.py

Access the Web Interface: After running the command, Streamlit will start a local web server. Open your web browser and navigate to http://localhost:8501.

Input House Details: Fill in the details for the house you want to predict the price for using the provided input fields.

Predict Price: Click the "Predict Price" button to see the predicted housing price.

Visualizations
Upon running the application, the model will also generate visualizations showing:

Actual vs. Predicted housing prices.
Distribution of residuals to assess model performance.


Educational Purpose
This project is designed for educational purposes to demonstrate the following concepts:

Data loading and preprocessing techniques.
Implementation of a Linear Regression model using TensorFlow.
Evaluation metrics for regression models.


Acknowledgements
Kaggle House Prices Dataset
TensorFlow Documentation
Streamlit Documentation