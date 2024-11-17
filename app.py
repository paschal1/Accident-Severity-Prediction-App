import streamlit as st
import pickle

# Load the model
with open("rf_model_p_pickle.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Streamlit app interface
st.title("Accident Severity Prediction")

# Input fields for the required features
number_of_vehicles = st.number_input("Number of Vehicles", min_value=1, max_value=100, step=1)
year = st.number_input("Year", min_value=2000, max_value=2100, step=1)

# When the user clicks the 'Predict' button
if st.button("Predict"):
    # Prepare input data as a list
    input_data = [number_of_vehicles, year]
    
    try:
        # Predict accident severity
        prediction = loaded_model.predict([input_data])
        st.write(f"Predicted Accident Severity: {prediction[0]}")
    except ValueError as e:
        st.error(f"Error: {e}")
