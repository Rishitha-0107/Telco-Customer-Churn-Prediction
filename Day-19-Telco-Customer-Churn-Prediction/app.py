import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# 1. Load Dataset
# -------------------------------
st.title("📞 Customer Churn Prediction App")

@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# 2. Understand Customer Attributes
# -------------------------------
st.subheader("Dataset Information")
st.write(df.info())

# -------------------------------
# 3. Data Cleaning
# -------------------------------
df = df.drop("customerID", axis=1)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# Encode target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Encode categorical features
cat_cols = df.select_dtypes(include="object").columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# -------------------------------
# 4. Feature Selection
# -------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

# -------------------------------
# 5. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 6. Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 7. Train Model (Logistic Regression)
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------
# 8. Model Evaluation
# -------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.subheader("Model Performance")
st.write("**Accuracy:**", accuracy)

# Confusion Matrix Plot
fig, ax = plt.subplots()
ax.imshow(cm, cmap="Blues")
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig)

# -------------------------------
# 9. Business Analysis
# -------------------------------
TN, FP, FN, TP = cm.ravel()

st.subheader("Churn Analysis")
st.write("✔ Correctly identified churn customers (TP):", TP)
st.write("❌ Non-churn customers misclassified as churn (FP):", FP)

# -------------------------------
# 10. Predict for New Customer
# -------------------------------
st.subheader("Predict Churn for New Customer")

input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(col, float(df[col].min()), float(df[col].max()))

input_df = pd.DataFrame([input_data])
input_df = scaler.transform(input_df)

prediction = model.predict(input_df)
probability = model.predict_proba(input_df)[0][1]

if st.button("Predict Churn"):
    if prediction[0] == 1:
        st.error(f"⚠ Likely to Churn (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Likely to Stay (Probability: {1 - probability:.2f})")
