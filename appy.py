import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Load the pre-trained model
with open('rf_model_pickle.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Load feature importances
with open('feature_importances.pkl', 'rb') as f:
    feature_importances = pickle.load(f)

# Initialize LabelEncoder (to match preprocessing)
encoder = LabelEncoder()

# Define function to encode categorical columns
def encode_input_data(data):
    categorical_columns = ['Road_Type', 'Junction_Control', 'Light_Conditions', 
                           'Weather_Conditions', 'Road_Surface_Conditions', 
                           'Urban_or_Rural_Area']
    for col in categorical_columns:
        data[col] = encoder.fit_transform(data[col])
    return data

# Streamlit app interface
st.title("Accident Severity Prediction")

# Add user input fields
road_type = st.selectbox("Road Type", ['Urban', 'Rural'])
speed_limit = st.number_input("Speed Limit", min_value=10, max_value=100, step=1)
junction_control = st.selectbox("Junction Control", ['Controlled', 'Uncontrolled'])
light_conditions = st.selectbox("Light Conditions", ['Day', 'Night'])
weather_conditions = st.selectbox("Weather Conditions", ['Clear', 'Fog', 'Rain'])
road_surface_conditions = st.selectbox("Road Surface Conditions", ['Dry', 'Wet'])
urban_or_rural_area = st.selectbox("Urban or Rural Area", ['Urban', 'Rural'])
year = st.number_input("Year", min_value=2000, max_value=2024, step=1)

# Collect input data
input_data = [
    road_type, speed_limit, junction_control, light_conditions,
    weather_conditions, road_surface_conditions, urban_or_rural_area, year
]

# Encode categorical variables (same encoding as in training)
encoded_data = [
    0 if road_type == 'Urban' else 1,
    speed_limit,
    0 if junction_control == 'Controlled' else 1,
    0 if light_conditions == 'Day' else 1,
    0 if weather_conditions == 'Clear' else (1 if weather_conditions == 'Fog' else 2),
    0 if road_surface_conditions == 'Dry' else 1,
    0 if urban_or_rural_area == 'Urban' else 1,
    year
]

# Convert encoded data into a dataframe for prediction
# Ensure the columns match exactly with the model's expected columns
input_df = pd.DataFrame([encoded_data], columns=['Road_Type', 'Speed_limit', 'Junction_Control', 
                                                 'Light_Conditions', 'Weather_Conditions', 
                                                 'Road_Surface_Conditions', 'Urban_or_Rural_Area', 'Year'])

# Ensure that no extra columns are included in the input DataFrame
input_df = input_df.drop(columns=['Unnamed: 0'], errors='ignore')  # Remove 'Unnamed: 0' if it exists

# When the user clicks the 'Predict' button
if st.button("Predict"):
    # Predict the accident severity
    try:
        prediction = loaded_model.predict(input_df)

        # Show the prediction result
        severity_mapping = {0: "Low Severity", 1: "Medium Severity", 2: "High Severity"}  # Example mapping
        st.write(f"Predicted Accident Severity: {severity_mapping.get(int(prediction[0]), 'Unknown')}")
    except ValueError as e:
        st.error(f"Error during prediction: {e}")

# Optional: Visualize feature importance (for tree-based models)
st.subheader("Feature Importance for Random Forest Model")

# Ensure feature importances are available
if feature_importances is not None:
    # Visualizing feature importance (sorted in descending order)
    feature_importances_sorted = sorted(zip(feature_importances, input_df.columns), reverse=True)
    sorted_importances = [item[0] for item in feature_importances_sorted]
    sorted_columns = [item[1] for item in feature_importances_sorted]

    # Create figure and axes for the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot the feature importances
    sns.barplot(x=sorted_importances, y=sorted_columns, ax=ax)
    ax.set_title("Feature Importances - Random Forest")

    # Display the plot in Streamlit
    st.pyplot(fig)
