import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


# Function to load the dataset
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("Error: The specified file was not found.")
        return None


# Function for data preprocessing
def preprocess_data(data):
    # Handle missing values
    # Impute numerical features with mean
    num_cols = data.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        data[col].fillna(data[col].mean(), inplace=True)

    # Impute categorical features with mode
    cat_cols = data.select_dtypes(include=[object]).columns
    for col in cat_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # Normalize numerical features
    scaler = StandardScaler()
    data[num_cols] = scaler.fit_transform(data[num_cols])

    # Convert categorical variables using target encoding
    for col in cat_cols:
        target_mean = data.groupby(col)['SalePrice'].mean()
        data[col] = data[col].map(target_mean)

    return data


# Function to split data into training and testing sets
def split_data(data):
    X = data.drop('SalePrice', axis=1)
    y = data['SalePrice']
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Function to build and train the model
def build_and_train_model(X_train, y_train):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer for regression
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, verbose=0)
    return model


# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_pred.flatten() - y_test) ** 2))
    return rmse, y_pred


# Function to visualize results
def visualize_results(y_test, y_pred):
    plt.figure(figsize=(12, 6))

    # Actual vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Housing Prices')

    # Residual plot
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred.flatten()
    sns.histplot(residuals, bins=30, kde=True)
    plt.title('Residuals Distribution')
    plt.xlabel('Residuals')

    plt.tight_layout()
    plt.show()


# Main function to run the program
def main():
    # Load data
    data = load_data('house_prices.csv')
    if data is None:
        return

    # Preprocess data
    data = preprocess_data(data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(data)

    # Build and train the model
    model = build_and_train_model(X_train, y_train)

    # Evaluate the model
    rmse, y_pred = evaluate_model(model, X_test, y_test)
    print(f"RMSE: {rmse}")

    # Visualize results
    visualize_results(y_test, y_pred)

    # Streamlit GUI for predictions
    st.title("Housing Price Prediction")
    st.write("Enter house details to predict the price.")

    # Input fields for user input
    input_data = {}
    for col in X_train.columns:
        input_data[col] = st.number_input(f"Enter {col}", value=0.0)

    if st.button("Predict Price"):
        input_df = pd.DataFrame([input_data])
        input_df[num_cols] = scaler.transform(input_df[num_cols])  # Normalize numerical features
        for col in cat_cols:
            input_df