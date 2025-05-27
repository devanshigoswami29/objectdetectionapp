import os
os.environ['YOLO_CONFIG_DIR'] = '/tmp'

import streamlit as st
from PIL import Image
import torch

st.title("Object Detection App (YOLOv5)")

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True, force_reload=True)
    results = model(image)
    results.render()
    st.image(results.ims[0], caption="Detection Result", use_column_width=True)
