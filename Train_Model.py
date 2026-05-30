import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv(r"D:\Customer Churn Project\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop customer ID
df.drop("customerID", axis=1, inplace=True)

# Clean TotalCharges
df["TotalCharges"] = df["TotalCharges"].replace(" ", None)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Encode categorical columns
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Save columns
joblib.dump(X.columns.tolist(), "columns.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Save model
joblib.dump(model, "customer_churn_model.pkl")

print("Model saved successfully!")