import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('vehicles.csv')

# Drop irrelevant columns (e.g., 'id', 'url', 'region_url', 'VIN', 'image_url', 'description', 'county', 'posting_date')
df = df.drop(['id', 'url', 'region_url', 'VIN', 'image_url', 'description', 'county', 'posting_date', 'lat', 'long'], axis=1)

# Handle missing values if necessary
df = df.dropna()

# Convert categorical columns to numerical using Label Encoding
le = LabelEncoder()
categorical_columns = ['region', 'manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission', 'drive', 'size', 'type', 'paint_color', 'state']
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Separate features (X) and target variable (y)
X = df.drop('price', axis=1)
y = df['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
rf_model.fit(X_train, y_train)

# Get feature importances
feature_importances = rf_model.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 8))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance Analysis')
plt.show()

# Display the sorted feature importances
print("Feature Importances:")
print(feature_importance_df)
