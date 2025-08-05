import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv(r"C:\Users\nemun\Downloads\Customer-deposit-prediction-main\Customer-deposit-prediction-main\Bank_ml_project_balanced_50_50.csv")

# Select important features + target
features = ["job", "balance", "duration", "campaign", "pdays", "previous", "poutcome"]
target = "deposit"
X = df[features]
y = df[target]

# Encode categorical features
label_encoders = {}
for col in ["job", "poutcome"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
numerical_features = ["balance", "duration", "campaign", "pdays", "previous"]
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train smaller Random Forest model
rf_model = RandomForestClassifier(n_estimators=4, random_state=42)
rf_model.fit(X_train, y_train)

# ✅ Save compressed model
with open("random_forest.pkl", "wb") as model_file:
    pickle.dump(rf_model, model_file)

# ✅ Save label encoders and scaler
with open("label_encoders.pkl", "wb") as le_file:
    pickle.dump(label_encoders, le_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("✅ Model (rfm.pkl), Label Encoders, and Scaler saved successfully!")
