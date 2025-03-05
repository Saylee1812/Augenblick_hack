# import streamlit as st
# import pandas as pd
# import joblib
# import plotly.express as px
# from tensorflow.keras.models import load_model
# import numpy as np
# from sklearn.preprocessing import StandardScaler

# # Load dataset
# @st.cache_data
# def load_data():
#     df = pd.read_csv("data/test_FD001.txt", sep="\\s+")
#     df.columns = ["engine", "cycle"] + [f"sensor_{i}" for i in range(1, df.shape[1] - 1)]
#     return df

# df = load_data()

# # Load trained models
# rf_model = joblib.load("models/random_forest.pkl")
# knn_model = joblib.load("models/knn_model.pkl")
# svm_model = joblib.load("models/svm_model.pkl")
# lstm_model = load_model("models/lstm_model.h5")

# # Scale data (Important for ML models)
# scaler = StandardScaler()
# df_scaled = df.copy()
# df_scaled.iloc[:, 2:] = scaler.fit_transform(df.iloc[:, 2:])

# # Streamlit UI
# st.title("🚀 NASA Turbofan Engine Monitoring Dashboard")

# # Select Engine ID
# engine_ids = df["engine"].unique()
# selected_engine = st.selectbox("Select Engine:", engine_ids)

# # Filter data for selected engine
# engine_data = df[df["engine"] == selected_engine]
# engine_data_scaled = df_scaled[df_scaled["engine"] == selected_engine]  # Scaled version

# # Select Sensor to Plot
# sensor_cols = [col for col in df.columns if "sensor" in col]
# selected_sensor = st.selectbox("Select Sensor:", sensor_cols)

# # Plot Line Chart for Selected Sensor
# fig = px.line(engine_data, x="cycle", y=selected_sensor, title=f"{selected_sensor} Over Time")
# st.plotly_chart(fig)

# # Select Model for Anomaly Detection
# st.subheader("🔮 Predict Anomalies Using Machine Learning Models")
# model_choice = st.radio("Select Model:", ["Random Forest", "KNN", "SVM", "LSTM"])

# # Predict Anomalies
# if st.button("Predict Anomaly"):
#     # Extract latest sensor readings
#     latest_data = engine_data_scaled.iloc[-1, 2:].values.reshape(1, -1)

#     # Debug: Print input data
#     st.write("🔍 Model Input Data:", latest_data)

#     # Perform prediction
#     if model_choice == "Random Forest":
#         prediction = rf_model.predict(latest_data)[0]
#     elif model_choice == "KNN":
#         prediction = knn_model.predict(latest_data)[0]
#     elif model_choice == "SVM":
#         prediction = svm_model.predict(latest_data)[0]
#     elif model_choice == "LSTM":
#         latest_data = latest_data.reshape(1, latest_data.shape[1], 1)  # Reshape for LSTM
#         prediction = lstm_model.predict(latest_data)[0][0]

#     # Debug: Show raw prediction score
#     st.write(f"🔮 Raw Prediction Score: {prediction}")

#     # Dynamic Thresholding
#     anomaly_threshold = 0.02  # Lowered from 0.3 to 0.02
#     if prediction > anomaly_threshold:
#      st.error(f"⚠️ Anomaly Detected! Score: {round(prediction, 4)}")
#     else:
#      st.success(f"✅ Normal Operation. Score: {round(prediction, 4)}")

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler

# ✅ Load Data
@st.cache_data
def load_data():
    col_names = ['engine', 'cycle', 'setting_1', 'setting_2', 'setting_3',
                 "Fan Inlet Temp", "LPC Outlet Temp", "HPC Outlet Temp", "LPT Outlet Temp",
                 "Fan Inlet Pressure", "Bypass Duct Pressure", "HPC Outlet Pressure",
                 "Physical Fan Speed", "Physical Core Speed", "Engine Pressure Ratio",
                 "HPC Outlet Static Pressure", "Fuel Flow Ratio", "Corrected Fan Speed",
                 "Corrected Core Speed", "Bypass Ratio", "Burner Fuel-Air Ratio",
                 "Bleed Enthalpy", "Required Fan Speed", "Required Fan Conversion Speed",
                 "High-Pressure Turbine Cool Air Flow", "Low-Pressure Turbine Cool Air Flow"]
    
    df_train = pd.read_csv("data/train_FD001.txt", sep='\\s+', header=None, names=col_names)
    df_test = pd.read_csv("data/test_FD001.txt", sep='\\s+', header=None, names=col_names)
    df_test_RUL = pd.read_csv("data/RUL_FD001.txt", sep='\\s+', header=None, names=['RUL'])
    
    return df_train, df_test, df_test_RUL

df_train, df_test, df_test_RUL = load_data()

# ✅ Model Performance Data
model_performance = pd.DataFrame({
    'Model': ['kNN', 'SVM', 'Random Forest', 'LSTM'],
    'R²': [0.7979, 0.7813, 0.8160, 0.7834],
    'RMSE': [18.68, 19.43, 17.83, 19.34],
    'Training Time (s)': [0.063, 31.368, 5.872, 26.283],
    'Prediction Time (s)': [0.056, 0.288, 0.129, 1.157],
    'Total Time (s)': [0.119, 31.657, 6.001, 27.440]
})

# ✅ Load ML Models
rf_model = joblib.load("models/random_forest.pkl")
knn_model = joblib.load("models/knn_model.pkl")
svm_model = joblib.load("models/svm_model.pkl")
lstm_model = load_model("models/lstm_model.h5")

# ✅ Scale Data for Anomaly Detection
scaler = StandardScaler()
df_test_scaled = df_test.copy()
df_test_scaled.iloc[:, 2:] = scaler.fit_transform(df_test.iloc[:, 2:])

# ✅ Streamlit UI
st.title("🚀 Turbofan Engine Predictive Maintenance Dashboard")
st.sidebar.header("Navigation")
view_option = st.sidebar.selectbox("Choose a View", ["Dataset", "Model Performance", "Predictions", "Anomaly Detection"])

# ✅ Dataset Exploration
if view_option == "Dataset":
    st.subheader("Training Data Preview")
    st.write(df_train.head())

    # Feature Selection for Visualization
    feature = st.selectbox("Select Feature to Visualize", df_train.columns[2:])
    fig = px.histogram(df_train, x=feature, title=f"Distribution of {feature}")
    st.plotly_chart(fig)

# ✅ Model Performance Comparison
elif view_option == "Model Performance":
    st.subheader("Model Performance Comparison")
    st.write(model_performance)

    # Model Performance Graphs
    fig = px.bar(model_performance, x='Model', y='R²', title="R-Squared Comparison", color='Model')
    st.plotly_chart(fig)

    fig = px.bar(model_performance, x='Model', y='RMSE', title="RMSE Comparison", color='Model')
    st.plotly_chart(fig)

    fig = px.bar(model_performance, x='Model', y='Total Time (s)', title="Total Computation Time", color='Model')
    st.plotly_chart(fig)

# ✅ Remaining Useful Life (RUL) Visualization
elif view_option == "Predictions":
    st.subheader("Remaining Useful Life (RUL) Predictions")
    st.write(df_test_RUL.head())

    fig = px.histogram(df_test_RUL, x='RUL', title="RUL Distribution")
    st.plotly_chart(fig)

# ✅ Anomaly Detection
elif view_option == "Anomaly Detection":
    st.subheader("🔍 Engine Anomaly Detection")

    # Select Engine ID
    engine_ids = df_test["engine"].unique()
    selected_engine = st.selectbox("Select Engine:", engine_ids)

    # Filter Data for Selected Engine
    engine_data = df_test[df_test["engine"] == selected_engine]
    engine_data_scaled = df_test_scaled[df_test_scaled["engine"] == selected_engine]

    # Select Sensor for Visualization
    sensor_cols = df_test.columns[2:]
    selected_sensor = st.selectbox("Select Sensor:", sensor_cols)

    # Plot Sensor Trends
    fig = px.line(engine_data, x="cycle", y=selected_sensor, title=f"{selected_sensor} Over Time")
    st.plotly_chart(fig)

    # Select Model for Prediction
    st.subheader("🔮 Predict Anomalies Using Machine Learning Models")
    model_choice = st.radio("Select Model:", ["Random Forest", "KNN", "SVM", "LSTM"])

    # Predict Anomalies
    if st.button("Predict Anomaly"):
        latest_data = engine_data_scaled.iloc[-1, 2:].values.reshape(1, -1)

        # Debug: Print Input Data
        st.write("🔍 Model Input Data:", latest_data)

        # Perform Prediction
        if model_choice == "Random Forest":
            prediction = rf_model.predict(latest_data)[0]
        elif model_choice == "KNN":
            prediction = knn_model.predict(latest_data)[0]
        elif model_choice == "SVM":
            prediction = svm_model.predict(latest_data)[0]
        elif model_choice == "LSTM":
            latest_data = latest_data.reshape(1, latest_data.shape[1], 1)
            prediction = lstm_model.predict(latest_data)[0][0]

        # Debug: Show Raw Prediction Score
        st.write(f"🔮 Raw Prediction Score: {prediction}")

        # Thresholding for Anomaly Detection
        anomaly_threshold = 0.02  # Lowered from 0.3 to 0.02
        if prediction > anomaly_threshold:
            st.error(f"⚠️ Anomaly Detected! Score: {round(prediction, 4)}")
        else:
            st.success(f"✅ Normal Operation. Score: {round(prediction, 4)}")
