# AI-powered-Agricultural-Yield-Prediction
Utilize ML and satellite imagery to predict crop yields and optimize resource allocation for farmers in developing countries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import rasterio
from rasterio.plot import show

# Example dataset paths
# Replace these paths with the actual paths to your satellite imagery and yield data
satellite_image_path = 'path/to/satellite/image.tif'
yield_data_csv_path = 'path/to/yield_data.csv'

# Load yield data
yield_data = pd.read_csv(yield_data_csv_path)
# Assume yield_data has columns: ['Region', 'Year', 'Yield']

# Function to extract features from satellite images
def extract_features(image_path):
    with rasterio.open(image_path) as src:
        # Read the entire array
        image_data = src.read()
        # Calculate mean and standard deviation as simple features for demonstration
        mean_features = np.mean(image_data, axis=(1, 2))
        std_features = np.std(image_data, axis=(1, 2))
    return np.concatenate([mean_features, std_features])

# Example: Extract features from satellite imagery
features = extract_features(satellite_image_path)

# Prepare the dataset for training
# This is a simplified example assuming each row in the yield_data corresponds to the features extracted from the images
X = np.array([features for _ in range(len(yield_data))])  # Placeholder for actual feature extraction
y = yield_data['Yield'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae:.2f}')

# Example visualization - Plotting the importance of features
feature_importances = model.feature_importances_
plt.barh(range(len(feature_importances)), feature_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Feature Index')
plt.show()

# Note: This is a simplified demo. In a real-world scenario, you would:
# 1. Use more sophisticated methods for feature extraction from satellite images.
# 2. Have a more complex dataset linking specific images to yield data.
# 3. Potentially use deep learning models for direct image analysis.
# 4. Consider environmental and temporal factors in your analysis.
