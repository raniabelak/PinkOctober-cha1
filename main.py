import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load data
train_data = pd.read_csv('data/train.csv')
test1_data = pd.read_csv('data/test.csv')

# Check for missing values
print(train_data.isnull().sum())
print(test1_data.isnull().sum())

# Prepare features and target
X_train = train_data.drop(columns=['id', 'diagnosis'])
y_train = train_data['diagnosis']  # Keep 'M' and 'B' as is

# Prepare test data
X_test1 = test1_data.drop(columns=['id'])

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test1_scaled = scaler.transform(X_test1)

# Create and train the model
model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)

# Make predictions on test data
predictions1 = model.predict(X_test1_scaled)

# Create DataFrame with predictions
predictions1_df = pd.DataFrame({
    'id': test1_data['id'],
    'predicted_diagnosis': predictions1
})

# Save predictions to CSV
predictions1_df.to_csv('data/predictions1_df.csv', index=False)
