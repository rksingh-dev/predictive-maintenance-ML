import streamlit as st
import joblib
import pandas as pd

rfc = joblib.load('model.joblib')

st.title("Machine Predictive Maintenance Classification")

#  Type
#  Air temperature [K]
#  Process temperature [K]
#  Rotational speed [rpm]
#  Torque [Nm]	
#  Tool wear [min]

# getting the input data from the user
col1, col2 = st.columns(2)

with col1:
    selected_type = st.selectbox('Select a Type', ['Low', 'Medium', 'High'])
    type_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    selected_type = type_mapping[selected_type]

  
with col2:
    air_temperature = st.number_input('Air temperature [K]', min_value=290.0, max_value=310.0, value=298.0, step=0.1)

with col1:
    process_temperature = st.number_input('Process temperature [K]', min_value=300.0, max_value=320.0, value=308.0, step=0.1)

with col2:
    rotational_speed = st.number_input('Rotational speed [rpm]', min_value=1000, max_value=3000, value=1500, step=10)

with col1:
    torque = st.number_input('Torque [Nm]', min_value=0.0, max_value=100.0, value=40.0, step=0.1)

with col2:
    tool_wear = st.number_input('Tool wear [min]', min_value=0, max_value=300, value=0, step=1)


# code for Prediction
failure_pred = ''

# Define the failure type categories (same order as in the training notebook)
failure_types = ['No Failure', 'Heat Dissipation Failure', 'Power Failure', 
                 'Overstrain Failure', 'Tool Wear Failure', 'Random Failures']

# creating a button for Prediction

if st.button('Predict Failure'):
    # Create a DataFrame with proper feature names to avoid warnings
    input_data = pd.DataFrame([[selected_type, air_temperature, 
                                process_temperature, rotational_speed,
                                torque, tool_wear]], 
                              columns=['Type', 'Air temperature [K]', 'Process temperature [K]',
                                       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])
    
    prediction = rfc.predict(input_data)
    
    # Get the failure type name from the prediction
    failure_pred = failure_types[prediction[0]]
    
    # Show prediction with appropriate color
    if prediction[0] == 0:
        st.success(f"✅ Prediction: {failure_pred}")
    else:
        st.error(f"⚠️ Prediction: {failure_pred}")