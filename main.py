import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Step 1: Load Dataset
df = pd.read_excel("dataset\customer_churn_sample.xlsx")

# Step 2: Data Preprocessing
df.drop(columns=["CustomerID"], inplace=True)

# Encode categorical columns
label_encoders = {}
categorical_cols = ["Gender", "Payment_Method", "Contract_Type"]

for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Handle missing values
df.fillna(df.median(), inplace=True)

# Scale numerical columns
scaler = StandardScaler()
num_cols = ["Age", "Subscription_Length_Months", "Monthly_Bill", "Total_Charges"]
df[num_cols] = scaler.fit_transform(df[num_cols])

# Step 3: Split Data
X = df.drop(columns=["Churn"])
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train ML Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Step 6: Save Model
with open("churn_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as churn_model.pkl")
