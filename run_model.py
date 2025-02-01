import pandas as pd
from xgboost_model_pipeline import Model
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('heart_dataset.csv')

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Initialize and train the model
model = Model(train_df, test_df)
model.process_data()
model.train()
model.predict()
model.eval() 