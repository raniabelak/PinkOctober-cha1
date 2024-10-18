import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('data/train.csv')
test1_data = pd.read_csv('data/test.csv')

print(train_data.isnull().sum())
print(test1_data.isnull().sum())

X = train_data.drop(columns=['id', 'diagnosis'])
y = train_data['diagnosis'].map({'B': 0, 'M': 1}).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 

X_test1 = test1_data.drop(columns=['id'])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_test1_scaled = scaler.transform(X_test1)

model = LogisticRegression(max_iter=2000)

model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)
predictions1 = model.predict(X_test1_scaled)

predictions1_df = pd.DataFrame({
    'id': test1_data['id'],
    'predicted_diagnosis': predictions1
})

predictions1_df.to_csv('data/predictions.csv', index=False)

accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')