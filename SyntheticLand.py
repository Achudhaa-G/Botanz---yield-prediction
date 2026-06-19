import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_regression

# Create a synthetic dataset
def create_synthetic_data(n_samples=1000):
    np.random.seed(42)
    locations = ['Location_A', 'Location_B', 'Location_C', 'Location_D']
    data = {
        'location': np.random.choice(locations, size=n_samples),
        'area': np.random.uniform(500, 5000, size=n_samples),
        'bedrooms': np.random.randint(1, 6, size=n_samples),
        'bathrooms': np.random.randint(1, 4, size=n_samples),
        'amenities': np.random.randint(0, 10, size=n_samples),
        'price': np.random.uniform(100000, 1000000, size=n_samples)
    }
    return pd.DataFrame(data)

# Function to automate preprocessing and training
def preprocess_and_train(data, features, target):
    # Separate features and target
    X = data[features]
    y = data[target]

    # Define numerical and categorical columns
    numerical_features = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
    categorical_features = [col for col in X.columns if X[col].dtype == 'object']

    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessors in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Define the complete pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model using the pipeline
    model_pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = model_pipeline.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    return model_pipeline

# Generate synthetic dataset
data = create_synthetic_data()

# Define features and target
features = ['location', 'area', 'bedrooms', 'bathrooms', 'amenities']
target = 'price'

# Call the automated preprocessing and training function
model_pipeline = preprocess_and_train(data, features, target)

# Visualization function
def visualize_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Prices')
    plt.show()

# Save the pipeline (optional)
import joblib
joblib.dump(model_pipeline, 'land_price_model_pipeline.pkl')

# Load the pipeline (for future use)
# model_pipeline = joblib.load('land_price_model_pipeline.pkl')
