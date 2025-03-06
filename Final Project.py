import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Drug Cases Predictor", layout="wide", page_icon="🚨")

# CSS for Modern UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-color: #2e3b4e;
        color: white;
    }
    h1 {
        color: #ff4b4b;
        text-align: center;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚨 Drug Cases Predictor App")
st.write("📌 Track drug cases and predict future crimes using Machine Learning")

url = "https://www.data.gov.in/backend/dms/v1/ogdp/resource/download/603189971/json/eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJkYXRhLmdvdi5pbiIsImF1ZCI6ImRhdGEuZ292Lmlu..."

@st.cache_data
def load_data():
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        if 'data' in data and isinstance(data['data'], list):
            return pd.DataFrame(data['data'])
    st.error("Failed to fetch data")
    return pd.DataFrame()

def preprocess_data(df):
    data = pd.concat([df.iloc[:, 2:13].astype(float), df.iloc[:, 13:24].astype(float)], axis=1)
    data.fillna(data.mean(), inplace=True)
    return data

df = load_data()
if not df.empty:
    data = preprocess_data(df)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data", "Predictions", "Future Forecast"])

    if page == "Data":
        st.subheader("📌 Drug Cases Data")
        st.dataframe(data)
        fig = px.line(data, x=data.index, y=data.mean(axis=1), title="Drug Cases Over Time")
        st.plotly_chart(fig)

    elif page == "Predictions":
        st.subheader("Machine Learning Model Training")
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor(n_estimators=200, random_state=42))])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"✅ MSE: {mse:.2f}")
        st.write(f"🔥 R² Score: {r2:.2f}")

        fig = px.scatter(x=y_test, y=y_pred, title="Actual vs Predicted Cases", labels={"x": "Actual", "y": "Predicted"})
        st.plotly_chart(fig)

    elif page == "Future Forecast":
        st.sidebar.title("🔮 Future Prediction")
        year = st.sidebar.slider("Select Year", 2023, 2030, 2023)
        latest_data = X.iloc[-1].values.reshape(1, -1)
        future_pred = pipeline.predict(latest_data)[0]

        st.write(f"📌 Predicted Drug Cases in {year}: **{int(future_pred)}**")

st.sidebar.write("Made with ❤️ by AI Student")