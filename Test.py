import base64
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import io
from skimage import measure
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(page_title="Doctor Dashboard | MRI Tumor Detection", page_icon="🧠", layout="centered")

# --- Session State Initialization ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "show_scan_2d" not in st.session_state:
    st.session_state.show_scan_2d = False
if "show_scan_3d" not in st.session_state:
    st.session_state.show_scan_3d = False

# --- Custom CSS Styling ---
def set_background():
    with open(r"C:\Brain MRI Seg\image.avif", "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            font-family: 'Arial', sans-serif;
        }}
        .title {{ text-align: center; font-size: 40px; font-weight: bold; color: #fff; }}
        .subtitle {{ text-align: center; font-size: 22px; color: #fff; }}
        .stButton>button {{ background-color: #3498db; color: white; height: 50px; width:100%; border-radius:8px; font-size:18px; }}
        .stTextInput>div>div>input {{ border:1px solid #3498db; padding:10px; border-radius:8px; }}
        </style>
    """, unsafe_allow_html=True)

# --- Login Form ---
def login():
    set_background()
    st.markdown("<div class='title'>🧠 NEURONIX</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>MRI Brain Tumor Detection System</div>", unsafe_allow_html=True)
    st.markdown("""<style>[data-testid="stSidebar"]{display:none;}</style>""", unsafe_allow_html=True)

    with st.form("login_form"):
        st.write(" Doctor Login Portal")
        user = st.text_input("👨‍⚕ Doctor ID")
        pwd = st.text_input("🔒 Password", type="password")
        if st.form_submit_button("Login"):
            if user=="val" and pwd=="ros":
                st.session_state.logged_in=True
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")

# --- Dashboard After Login ---
def dashboard():
    st.title("👨‍⚕ Doctor's Dashboard")
    st.markdown("### Welcome back 🩺")
    with st.sidebar:
        st.title("Navigation")
        if st.button("🧠 Scan for 2D"):
            st.session_state.show_scan_2d=True
            st.session_state.show_scan_3d=False
            st.rerun()
        if st.button("🧠 Scan for 3D"):
            st.session_state.show_scan_2d=False
            st.session_state.show_scan_3d=True
            st.rerun()
    if not (st.session_state.show_scan_2d or st.session_state.show_scan_3d):
        st.markdown("👉 Choose a scan type from the sidebar.")

# --- 2D Detection Page ---
def scan_for_2d():
    st.title("Brain Tumor Detection (2D Scan)")
    model = load_model(r'C:\Brain MRI Seg\unet_brain_segmentation.h5')
    IMG_SIZE=(224,224)
    img_file = st.file_uploader("Upload a .tif Image", type=["tif"], key="i2d")
    mask_file= st.file_uploader("Upload True Mask (.tif) (Optional)", type=["tif"], key="m2d")
    if img_file:
        img = np.array(Image.open(img_file))
        inp = cv2.resize(img,IMG_SIZE)/255.0
        pred = model.predict(inp[np.newaxis,...])[0]
        mask = (pred>0.5).astype(np.uint8)*255
        if mask.ndim==3 and mask.shape[-1]==1:
            mask=mask[:,:,0]
        st.subheader("Original")
        st.image(img, channels="RGB")
        st.subheader("Predicted Mask")
        st.image(mask, clamp=True, channels="GRAY")
        if mask_file:
            true = np.array(Image.open(mask_file))
            true = cv2.resize(true,IMG_SIZE,interpolation=cv2.INTER_NEAREST)
            true_bin=(true>127).astype(np.uint8)*255
            acc=(mask.flatten()==true_bin.flatten()).mean()*100
            st.success(f"Pixel Accuracy: {acc:.2f}%")
        buf=io.BytesIO(); Image.fromarray(mask).save(buf,format="TIFF")
        st.download_button("Download Mask",buf.getvalue(),"pred_mask.tif","image/tiff")

# --- 3D Detection Page ---
def scan_for_3d():
    st.title("Brain Tumor Detection (3D Scan via TIF stacking)")
    img_file = st.file_uploader("Upload original 2D .tif slice", type=["tif"], key="i3d")
    mask_file = st.file_uploader("Upload corresponding 2D mask .tif", type=["tif"], key="m3d")
    slices = st.slider("Number of slices to stack", min_value=10, max_value=100, value=50, step=10)
    
    if img_file and mask_file:
        # Load and process images
        img2d = np.array(Image.open(img_file))
        mask2d = np.array(Image.open(mask_file))
        
        # Normalize and threshold the brain image to create a binary volume
        brain_img = cv2.normalize(img2d, None, 0, 255, cv2.NORM_MINMAX)
        _, brain_binary = cv2.threshold(brain_img, 10, 255, cv2.THRESH_BINARY)
        
        # Create 3D volumes with gradual changes between slices
        brain_volume = np.stack([brain_binary]*slices, axis=0)
        tumor_mask = np.stack([(mask2d > 127).astype(np.uint8)]*slices, axis=0)
        
        # Add some variation to make the brain volume more realistic
        for i in range(1, slices):
            brain_volume[i] = cv2.erode(brain_volume[i-1], np.ones((3,3), np.uint8), iterations=1)
            tumor_mask[i] = cv2.erode(tumor_mask[i-1], np.ones((3,3), np.uint8), iterations=1)
        
        # Create figure
        fig = go.Figure()
        
        # Brain mesh - using the binary volume
        try:
            verts, faces, _, _ = measure.marching_cubes(brain_volume, level=128)
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color='lightblue', opacity=0.2, name='Brain', showscale=False,
                lighting=dict(ambient=0.7, diffuse=0.3)
            ))
        except Exception as e:
            st.warning(f"Could not create brain mesh: {str(e)}")
        
        # Tumor mesh - only where tumor exists
        try:
            if np.any(tumor_mask):
                tumor_verts, tumor_faces, _, _ = measure.marching_cubes(tumor_mask, level=0.5)
                fig.add_trace(go.Mesh3d(
                    x=tumor_verts[:, 0], y=tumor_verts[:, 1], z=tumor_verts[:, 2],
                    i=tumor_faces[:, 0], j=tumor_faces[:, 1], k=tumor_faces[:, 2],
                    color='red', opacity=0.8, name='Tumor', showscale=False,
                    lighting=dict(ambient=0.3, diffuse=0.7)
                ))
        except Exception as e:
            st.warning(f"Could not create tumor mesh: {str(e)}")
        
        # Update layout for better visualization
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=0.8)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            height=600,
            title="3D Brain with Tumor Visualization"
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Calculate tumor statistics
        if np.any(tumor_mask):
            vox_count = tumor_mask.sum()
            st.markdown(f"**Tumor voxel count:** {int(vox_count)}")
            com = np.argwhere(tumor_mask > 0).mean(axis=0)
            st.markdown(f"**Center of mass:** Slice: {com[0]:.1f}, Row: {com[1]:.1f}, Col: {com[2]:.1f}")
            st.markdown(f"**Tumor volume:** {vox_count} voxels")
        else:
            st.success("No tumor detected in the scan")

# --- Run App ---
if not st.session_state.logged_in:
    login()
else:
    if st.session_state.show_scan_2d:
        scan_for_2d()
    elif st.session_state.show_scan_3d:
        scan_for_3d()
    else:
        dashboard()