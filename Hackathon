import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap
import streamlit as st
import warnings
import os

# Suppress warnings and logs
warnings.filterwarnings("ignore", message="missing ScriptRunContext!")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Ensure file_path is defined early
file_path = 'customer_data.csv'

# Function to write data into a CSV file
def write_csv(file_path, csv_data):
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)
    print(f"Data successfully written to {file_path}")

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check the path.")
        exit()

    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    return data

# Function to visualize the data
def visualize_data(data):
    sns.pairplot(data, hue="gender")
    plt.title("Pairplot of Customer Data")
    plt.show()

# Function to train the model with GridSearchCV
def train_model_with_gridsearch(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    return predictions

# Function to generate insights
def generate_insights(data):
    high_loyalty_customers = data[data['loyalty_score'] > 8]
    print(f"Number of high-loyalty customers: {len(high_loyalty_customers)}")
    return high_loyalty_customers

# Function to calculate and display feature importances
def display_feature_importances(model, X_train):
    if hasattr(model, "feature_importances_"):
        feature_importance = model.feature_importances_
        feature_names = X_train.columns
        plt.barh(feature_names, feature_importance)
        plt.title("Feature Importances")
        plt.show()
    else:
        print("Feature importances not available for this model.")

# Function to run Streamlit app
def run_streamlit_app(model, high_loyalty_customers):
    st.title("Customer Loyalty Prediction")
    st.write("This app predicts customer loyalty scores and provides insights.")

    # Input form
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Income", min_value=10000, max_value=100000, value=50000)
    gender = st.selectbox("Gender", ["Male", "Female"])
    purchase_frequency = st.number_input("Purchase Frequency", min_value=1, max_value=20, value=5)

    # Prepare input data
    input_data = pd.DataFrame({
        "age": [age],
        "income": [income],
        "purchase_frequency": [purchase_frequency],
        "gender_Male": [1 if gender == "Male" else 0]
    })

    # Predict loyalty score
    if st.button("Predict Loyalty Score"):
        prediction = model.predict(input_data)
        st.write(f"Predicted Loyalty Score: {prediction[0]:.2f}")

    # Display high-loyalty customers
    if st.button("Show High-Loyalty Customers"):
        st.write(high_loyalty_customers)

# Main code execution
if __name__ == "__main__":
    # Step 1: Write data to CSV
    csv_data = [
        ["age", "income", "loyalty_score", "gender", "purchase_frequency"],
        [25, 50000, 7, "Male", 5],
        [34, 60000, 9, "Female", 8],
        [45, 75000, 8, "Male", 6],
        [23, 48000, 6, "Female", 4],
        [35, 52000, 10, "Male", 9]
    ]
    write_csv(file_path, csv_data)

    # Step 2: Load and preprocess data
    data = load_and_preprocess_data(file_path)
    print(data.head())

    # Step 3: Visualize data
    visualize_data(data)

    # Step 4: Split data into train and test sets
    X = data.drop('loyalty_score', axis=1)
    X = pd.get_dummies(X, drop_first=True)
    y = data['loyalty_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    # Step 5: Train model with GridSearchCV
    model = train_model_with_gridsearch(X_train, y_train)

    # Step 6: Evaluate model
    predictions = evaluate_model(model, X_test, y_test)

    # Step 7: Display feature importances
    display_feature_importances(model, X_train)

    # Step 8: SHAP values and dependence plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.dependence_plot("income", shap_values, X_test, feature_names=X_test.columns)

    # Step 9: Generate insights
    print(data['loyalty_score'].value_counts())  # Debug loyalty score distribution
    high_loyalty_customers = generate_insights(data)

    # Step 10: Run Streamlit app (uncomment this to deploy the app)
    # Uncomment the following line to run the Streamlit app
    run_streamlit_app(model, high_loyalty_customers)
