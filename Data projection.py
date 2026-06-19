import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Automating the synthetic dataset creation and model training
class CropYieldPredictor:
    def __init__(self, dataset_path='synthetic_crop_data.csv', model_path='synthetic_crop_yield_model.pkl'):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.model = None

    def generate_synthetic_data(self, num_samples=1000):
        """Generates synthetic crop yield data and saves it to a CSV file."""
        synthetic_data = pd.DataFrame({
            'Temperature': np.random.uniform(15, 35, num_samples),
            'Rainfall': np.random.uniform(50, 200, num_samples),
            'Soil_Type': np.random.choice(['Loam', 'Clay', 'Sandy'], num_samples),
            'Crop_Type': np.random.choice(['Wheat', 'Rice', 'Corn'], num_samples),
            'Yield': np.random.uniform(200, 500, num_samples)
        })
        synthetic_data.to_csv(self.dataset_path, index=False)
        print(f"Synthetic dataset created and saved as '{self.dataset_path}'")

    def prepare_data(self):
        """Loads the dataset, preprocesses it, and splits it into training and testing sets."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file '{self.dataset_path}' not found.")

        data = pd.read_csv(self.dataset_path)
        data = pd.get_dummies(data, columns=['Soil_Type', 'Crop_Type'], drop_first=True)
        X = data.drop('Yield', axis=1)
        y = data['Yield']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        """Trains a Random Forest model and saves it to a file."""
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, self.model_path)
        print(f"Model trained and saved as '{self.model_path}'")

    def evaluate_model(self, X_test, y_test):
        """Evaluates the model on the testing data and prints metrics."""
        if not self.model:
            self.model = joblib.load(self.model_path)

        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Squared Error: {mse}")
        print(f"R^2 Score: {r2}")

    def automate(self):
        """End-to-end automation of dataset generation, model training, and evaluation."""
        self.generate_synthetic_data()
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.train_model(X_train, y_train)
        self.evaluate_model(X_test, y_test)

# Automate the workflow
if __name__ == "__main__":
    predictor = CropYieldPredictor()
    predictor.automate()
