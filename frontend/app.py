import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model_path = r"C:\Users\MUSKAAN\Documents\M.E Folder\Design Project\CropWaterStressApp\backend\model\best_model.pkl"
scaler_path = r"C:\Users\MUSKAAN\Documents\M.E Folder\Design Project\CropWaterStressApp\backend\model\scaler.pkl"

# Check if model and scaler files exist
if os.path.exists(model_path):
    print(f"Model file exists at: {os.path.abspath(model_path)}")
    model = joblib.load(model_path)  # Load the trained model
else:
    raise ValueError("Model file is missing. Please check the file path and ensure the model was saved correctly.")

if os.path.exists(scaler_path):
    print(f"Scaler file exists at: {os.path.abspath(scaler_path)}")
    scaler = joblib.load(scaler_path)  # Load the scaler
else:
    raise ValueError("Scaler file is missing. Please check the file path and ensure the scaler was saved correctly.")

# Example dataset (replace with actual dataset)
data = {
    'Temperature': [25, 30, 22, 35, 28, 26, 29, 31],
    'Humidity': [60, 50, 55, 40, 45, 52, 48, 50],
    'Soil Moisture': [30, 25, 35, 15, 20, 25, 30, 35],
    'Wind Speed': [5, 6, 7, 5, 6, 4, 5, 7],
    'Rainfall': [10, 20, 15, 5, 10, 8, 12, 20],
    'Water Stress Level': [0, 1, 0, 2, 1, 0, 1, 2]  # 0 = Low, 1 = Moderate, 2 = High
}

# Create dataframe
df = pd.DataFrame(data)

# Feature and Target variable
X = df.drop('Water Stress Level', axis=1)
y = df['Water Stress Level']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model (for illustration, assuming the model was trained earlier and saved)
# In reality, you would train the model before using it for prediction

# Streamlit layout
st.set_page_config(page_title="Crop Water Stress Detection", layout="wide")
st.markdown("""
    <style>
        .reportview-container {
            background-image: url('static/irrigation_background.jpg');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }
        .main {
            padding-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🌾 Crop Water Stress Detection")
st.markdown("""
This application helps you determine the water stress level of your crops based on key environmental factors.
""")

# Display default image of irrigation on the landing page
st.image("static/crop_background.jpg", caption="Irrigation System", use_container_width=True)

# Sidebar inputs for environmental data using sliders only
st.sidebar.header("📥 Enter Environmental Data")
temp = st.sidebar.slider("🌡️ Temperature (°C)", -10.0, 50.0, 25.0)
humidity = st.sidebar.slider("💧 Humidity (%)", 0.0, 100.0, 50.0)
soil_moisture = st.sidebar.slider("🌱 Soil Moisture (%)", 0.0, 100.0, 30.0)
wind_speed = st.sidebar.slider("💨 Wind Speed (m/s)", 0.0, 20.0, 5.0)
rainfall = st.sidebar.slider("🌧️ Rainfall (mm)", 0.0, 200.0, 10.0)

# Placeholder for the missing 6th feature (NDVI)
ndvi = st.sidebar.slider("🌿 NDVI (Normalized Difference Vegetation Index)", -1.0, 1.0, 0.5)

# Prediction Button
if st.sidebar.button("🌾 Predict Water Stress Level"):
    # Prediction based on user input (Including NDVI)
    user_input = np.array([[temp, humidity, soil_moisture, wind_speed, rainfall, ndvi]])
    
    # Scale the input data using the same scaler
    user_input_scaled = scaler.transform(user_input)  # Use the scaler object to scale the input data
    
    # Predict using the trained model
    prediction = model.predict(user_input_scaled)[0]

    # Define prediction result and recommendation
    if prediction == 0:
        stress_level = "Low"
        recommendation = "Irrigation not needed."
        crop_image = "static/low_stress.jpg"
    elif prediction == 1:
        stress_level = "Moderate"
        recommendation = "Consider irrigation within 24 hours."
        crop_image = "static/moderate_stress.jpg"
    else:
        stress_level = "High"
        recommendation = "Urgent irrigation required to prevent crop damage."
        crop_image = "static/high_stress.jpg"

    # Display Water Stress Level, Predicted Image, Recommendation, and Watering Tips in one section
    with st.container():
        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader(f"🌱 Water Stress Level: {stress_level}")
            st.write(f"🚨 **Recommendation**: {recommendation}")
            st.subheader("💧 Watering Tips")
            if prediction == 0:
                st.write("🌱 **Low Stress Level**: No irrigation needed. Check moisture levels periodically.")
            elif prediction == 1:
                st.write("🌱 **Moderate Stress Level**: Irrigate in the next 24 hours, check the weather forecast.")
            else:
                st.write("🌱 **High Stress Level**: Urgent irrigation needed to prevent crop damage. Ensure soil moisture retention.")

        with col2:
            st.image(crop_image, caption=f"Crop with {stress_level} stress", width=300)  # Resize the predicted image with width only

    # Footer Section
    st.markdown("---")
    footer = """
        <style>
            .footer {
                padding: 10px;
                background-color: #f1f1f1;
                text-align: center;
                font-size: 14px;
                color: #333;
            }
            .footer a {
                color: #0066cc;
                text-decoration: none;
            }
            .footer a:hover {
                text-decoration: underline;
            }
        </style>
        <div class="footer">
            <p>Developed with ❤️ by <strong><a href="https://www.linkedin.com/in/muskaan-mahajan-196578213/" target="_blank">Muskaan</a> & <a href="https://in.linkedin.com/in/jagrity-rana-24837021b" target="_blank">Jagrity</a></strong></p>
            <p>For any inquiries, reach out to us via LinkedIn or Email</p>
            <p>Powered by AI | Crop Water Stress Detection</p>
        </div>
    """
    st.markdown(footer, unsafe_allow_html=True)

else:
    st.markdown("Please enter the environmental data and click the 'Predict Water Stress Level' button.")
