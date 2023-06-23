import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
import warnings

warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

st.set_page_config(page_title="SmartCrop", page_icon="https://cdn.jsdelivr.net/gh/twitter/twemoji@master/assets/72x72/1f33f.png", layout='centered', initial_sidebar_state="collapsed")

def load_model(modelfile):
	loaded_model = pickle.load(open('../models/RandomForest.pkl', 'rb'))
	return loaded_model

def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:center;"> SmartCrop:Crop Recommendation 🌱 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    col = st.columns(1)[0]

    with col:
        st.subheader(" Find out the most suitable crop to grow in your farm 👨‍🌾")
        N = st.number_input("Nitrogen", 0,140)
        P = st.number_input("Phosporus", 5,145)
        K = st.number_input("Potassium", 5,205)
        temp = st.number_input("Temperature",8.825675,43.675493)
        humidity = st.number_input("Humidity in %", 14.258040,99.981876)
        ph = st.number_input("Ph", 3.504752,9.935091)
        rainfall = st.number_input("Rainfall in mm",20.211267,298.560117)

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1,-1)
        
        if st.button('Predict'):

            loaded_model = load_model(r'RandomForest.pkl')
            prediction = loaded_model.predict(single_pred)
            col.write('''
		    ## Results 🔍 
		    ''')
            col.success(f"{prediction.item().title()} are recommended for your farm.")
    #code for html ☘️ 🌾 🌳 👨‍🌾  🍃
    hide_menu_style = """
    <style>
    .block-container {padding: 2rem 1rem 3rem;}
    #MainMenu {visibility: hidden;}
    </style>
    """

hide_menu_style = """
        <style>
        .block-container {padding: 2rem 1rem 3rem;}
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
	main()
     