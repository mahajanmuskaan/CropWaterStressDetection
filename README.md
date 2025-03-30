**Crop Water Stress Detection Web App**
=======================================

**Overview**
------------

This web application is designed to predict crop water stress levels using environmental data such as temperature, humidity, soil moisture, wind speed, rainfall, and NDVI (Normalized Difference Vegetation Index). It uses machine learning models to predict whether crops are experiencing **low**, **moderate**, or **high** water stress. Based on these predictions, the app provides irrigation recommendations to help farmers manage their water resources efficiently.

**Features**
------------

*   **Interactive User Input**: Input environmental data (temperature, humidity, soil moisture, wind speed, rainfall, NDVI) through sliders.
    
*   **Prediction**: Predict the water stress level based on the input data.
    
*   **Irrigation Recommendations**: Get irrigation suggestions based on the predicted water stress level.
    
*   **Visualization**: Display images and insights related to the predicted water stress level.
    
*   **User-Friendly Interface**: The app provides an easy-to-use interface for farmers to make informed decisions on irrigation.
    

**Technologies Used**
---------------------

*   **Python**
    
*   **Streamlit** (Web app framework)
    
*   **Machine Learning Models**:
    
    *   Random Forest Classifier
        
    *   Gradient Boosting Classifier
        
    *   XGBoost
        
    *   Support Vector Machine (SVM)
        
    *   K-Nearest Neighbors (KNN)
        
    *   Decision Tree Classifier
        
*   **Scikit-learn** (For training models)
    
*   **Pandas & NumPy** (For data manipulation)
    
*   **Matplotlib** (For visualization)
    

**Installation**
----------------

1.  git clone https://github.com/mahajanmuskaan/CropWaterStressDetection.git
    
2.  cd crop-water-stress-detection
    
3.  python -m venv venv
    
4.  **Activate the virtual environment:**
    
    *   venv\\Scripts\\activate
        
    *   source venv/bin/activate
        
5.  pip install -r requirements.txt
    
6.  streamlit run app.pyThis will start the app and open it in your browser.
    

**How to Use**
--------------

1.  **Input Data**:Use the interactive sliders to input environmental data:
    
    *   Temperature (°C)
        
    *   Humidity (%)
        
    *   Soil Moisture (%)
        
    *   Wind Speed (m/s)
        
    *   Rainfall (mm)
        
    *   NDVI (Normalized Difference Vegetation Index)
        
2.  **Predict Water Stress**:Once the data is input, click on the "Predict Water Stress Level" button to receive a prediction on the crop's water stress level.
    
3.  **View Results**:The prediction will display whether the crop is under **low**, **moderate**, or **high** water stress. Additionally, you will receive irrigation recommendations:
    
    *   **Low Stress**: No irrigation needed.
        
    *   **Moderate Stress**: Irrigation needed within 24 hours.
        
    *   **High Stress**: Urgent irrigation required.
        
4.  **Visualization**:Visual representations of the predicted water stress and related information will be displayed to help you understand the results.
    

**Model Training & Data**
-------------------------

The model used for predictions was trained on historical environmental data, including temperature, humidity, soil moisture, wind speed, rainfall, and NDVI. The dataset was preprocessed by handling missing values, scaling features, and using **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the dataset. The models were then trained using techniques like Random Forest, XGBoost, Gradient Boosting, and others, and evaluated for the best performance.

    

**Contributing**
----------------

Feel free to fork the repository, create branches, and submit pull requests. Contributions to improve the accuracy of the models, the UI, or adding new features are highly appreciated.

**Future Improvements**
-----------------------

*   **Real-Time Data Integration**: Connect the app to live weather data and IoT sensors for real-time irrigation recommendations.
    
*   **Mobile App Version**: Develop a mobile app for better accessibility for farmers in rural areas.
    
*   **Additional Features**: Include more environmental factors (e.g., soil pH, solar radiation) for improved predictions.
