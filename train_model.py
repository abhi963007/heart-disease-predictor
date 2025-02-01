import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Read the dataset
df = pd.read_csv('heart_dataset.csv')

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

# Convert categorical variables to numeric
df['sex'] = (df['sex'] == 'Male').astype(int)

# Convert all columns to numeric
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill any remaining NaN values with column means
df = df.fillna(df.mean())

# Prepare features and target
X = df.drop('target', axis=1)
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = xgb.XGBClassifier(
    random_state=42,
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False
)

# Train the model
model.fit(X_train, y_train)

# Save the model
model.save_model('xgb_trained.ubj')

print("Model trained and saved successfully!") 