import streamlit as st
from PIL import Image
import numpy as np
import cv2
import requests
import json

st.header("Document Classifier")

st.text("Detects whether your Govt. ID is an Aadhar Card or a Pan Card.")

uploaded_files = st.file_uploader("Choose a file", type=['png','jpg'],accept_multiple_files=True)

if uploaded_files is not None:

    for file in uploaded_files:

        image = Image.open(file)
        img_arr = np.array(image)
        img_res = cv2.resize(img_arr,(128,128))

        json_data = json.dumps({'img':img_res.tolist()})

        response = requests.post("http://127.0.0.1:5000/predict",json_data)

        st.image(img_res)

        st.write(response.text)