import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import warnings

warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

st.set_page_config(page_title="AgriAI", page_icon="https://cdn.jsdelivr.net/gh/twitter/twemoji@master/assets/72x72/1f33f.png", layout='centered', initial_sidebar_state="collapsed")

def load_model(modelfile):
    loaded_model = pickle.load(open(modelfile, 'rb'))
    return loaded_model

def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:center;"> AgriAI: Intelligent Crop Recommendation 🌱 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    label_mapping = {'apple': 0, 'banana': 1, 'blackgram': 2, 'chickpea': 3, 'coconut': 4, 'coffee': 5,
                 'cotton': 6, 'grapes': 7, 'jute': 8, 'kidneybeans': 9, 'lentil': 10, 'maize': 11,
                 'mango': 12, 'mothbeans': 13, 'mungbean': 14, 'muskmelon': 15, 'orange': 16,
                 'papaya': 17, 'pigeonpeas': 18, 'pomegranate': 19, 'rice': 20, 'watermelon': 21}
    col = st.columns(1)[0]

    with col:
        st.subheader(" Find out the most suitable crop to grow in your farm 👨‍🌾")
        # Adjusted ranges based on realistic agricultural conditions
        N = st.number_input("Nitrogen (in ppm)", min_value=0, max_value=500)  # Typical range is 0 to 500 ppm
        P = st.number_input("Phosphorus (in ppm)", min_value=0, max_value=200)  # Typical range is 0 to 200 ppm
        K = st.number_input("Potassium (in ppm)", min_value=0, max_value=500)  # Typical range is 0 to 500 ppm
        temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0)  # Range is -10°C to 50°C
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0)  # Humidity is typically 0% to 100%
        ph = st.number_input("Soil pH", min_value=3.0, max_value=9.0)  # Most soils have pH between 3.0 and 9.0
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=5000.0)

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        if st.button('Predict'):
            loaded_model = load_model('model.pkl')
            prediction = loaded_model.predict(single_pred)
            col.write('''
            ## Results 🔍 
            ''')
            # Assuming prediction.item() gives the index
            predicted_label_index = prediction.item()

# Get the corresponding category label
            predicted_label = next(key for key, value in label_mapping.items() if value == predicted_label_index)

# Print the result
            col.success(f"{predicted_label} is recommended by the A.I for your farm.Thank you by AgriAI")


    hide_menu_style = """
    <style>
    .block-container {padding: 2rem 1rem 3rem;}
    #MainMenu {visibility: hidden;}
    </style>
    """

    st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
