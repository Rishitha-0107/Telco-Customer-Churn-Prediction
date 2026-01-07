import os
import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# ----------------------------------
# 1. Load Dataset
# ----------------------------------
st.title("📞 Customer Churn Prediction App")

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return pd.read_csv(data_path)

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ----------------------------------
# 2. Dataset Information
# ----------------------------------
st.subheader("📋 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Rows", df.shape[0])
    st.metric("Total Columns", df.shape[1])

with col2:
    st.metric("Numerical Columns", df.select_dtypes(include=["int64", "float64"]).shape[1])
    st.metric("Categorical Columns", df.select_dtypes(include="object").shape[1])

with col3:
    st.metric("Missing Values", df.isnull().sum().sum())
st.subheader("📊 Column Details")

column_info = pd.DataFrame({
    "Column Name": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Missing Values": df.isnull().sum(),
    "Unique Values": [df[col].nunique() for col in df.columns]
})

st.dataframe(column_info, use_container_width=True)


# ----------------------------------
# 3. Data Cleaning
# ----------------------------------
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# Encode target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Encode categorical features & store mappings
cat_cols = df.select_dtypes(include="object").columns
category_mappings = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    category_mappings[col] = le.classes_

# ----------------------------------
# 4. Feature Selection
# ----------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

# ----------------------------------
# 5. Train-Test Split
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------
# 6. Feature Scaling
# ----------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------
# 7. Train Model
# ----------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ----------------------------------
# 8. Model Evaluation
# ----------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"**Accuracy:** {accuracy:.2f}")

# Confusion Matrix
# Confusion Matrix (Smaller Size)
fig, ax = plt.subplots(figsize=(3, 3))  # 👈 control size here
ax.imshow(cm, cmap="Blues")

ax.set_title("Confusion Matrix", fontsize=12)
ax.set_xlabel("Predicted", fontsize=10)
ax.set_ylabel("Actual", fontsize=10)

labels = [["TN", "FP"], ["FN", "TP"]]
for i in range(2):
    for j in range(2):
        ax.text(
            j, i,
            f"{labels[i][j]}\n{cm[i, j]}",
            ha="center", va="center",
            fontsize=11
        )

st.pyplot(fig)


# ----------------------------------
# 9. Business Analysis
# ----------------------------------
TN, FP, FN, TP = cm.ravel()

st.subheader("📈 Business Insights")
st.write("✔ Correctly identified churn customers (TP):", TP)
st.write("❌ Non-churn customers misclassified as churn (FP):", FP)
st.write("✔ Correctly identified non-churn customers (TN):", TN)
st.write("❌ Missed churn customers (FN):", FN)

st.write("**Total Customers:**", len(df))
st.write("**Customers Who Stayed:**", (df["Churn"] == 0).sum())
st.write("**Customers Who Churned:**", (df["Churn"] == 1).sum())

# ----------------------------------
# 10. Churn Insight Charts
# ----------------------------------
st.subheader("📊 Churn Insights")

col1, col2 = st.columns(2)

with col1:
    churn_counts = df["Churn"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.bar(["Stayed", "Churned"], churn_counts, color=["green", "red"])
    ax1.set_title("Customer Churn Distribution")
    st.pyplot(fig1)

with col2:
    contract_churn = df.groupby("Contract")["Churn"].mean()
    fig2, ax2 = plt.subplots()
    ax2.bar(contract_churn.index, contract_churn.values)
    ax2.set_title("Churn Rate by Contract Type")
    plt.xticks(rotation=30)
    st.pyplot(fig2)

# ----------------------------------
# 11. Predict Churn for New Customer
# ----------------------------------
st.subheader("🔮 Predict Churn for New Customer")

input_data = {}

for col in X.columns:
    if col in category_mappings:
        input_data[col] = st.selectbox(
            col, category_mappings[col]
        )
    else:
        input_data[col] = st.number_input(
            col,
            min_value=float(df[col].min()),
            max_value=float(df[col].max()),
            value=float(df[col].mean())
        )

# Encode categorical inputs
for col in category_mappings:
    input_data[col] = list(category_mappings[col]).index(input_data[col])

input_df = pd.DataFrame([input_data])
input_df = scaler.transform(input_df)

prediction = model.predict(input_df)
probability = model.predict_proba(input_df)[0][1]

if st.button("Predict Churn"):
    if prediction[0] == 1:
        st.error(f"⚠ Likely to Churn (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Likely to Stay (Probability: {1 - probability:.2f})")



