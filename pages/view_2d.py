# view_2d.py

import streamlit as st
from PIL import Image

st.set_page_config(page_title="2D MRI Image Viewer", page_icon="🧠")

st.title("🧾 2D MRI Image Analysis")
st.subheader("View and Analyze 2D Brain Scans")

st.markdown("Upload a 2D MRI image to begin analysis:")

uploaded_file = st.file_uploader("Choose a .jpg or .png file", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
    st.success("Image loaded successfully!")

    # You can add segmentation logic here later
    st.info("🚧 Tumor detection logic will be implemented here.")
