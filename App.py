import base64
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import io
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(page_title="Doctor Dashboard | MRI Tumor Detection", page_icon="🧠", layout="centered")

# --- Session State Initialization ---
if 'show_2d' not in st.session_state:
    st.session_state.show_2d = False
if 'show_3d' not in st.session_state:
    st.session_state.show_3d = False
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

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
    st.markdown("""
        <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        st.write("Doctor Login Portal")
        user = st.text_input("👨‍⚕ Doctor ID")
        pwd = st.text_input("🔒 Password", type="password")
        if st.form_submit_button("Login"):
            if user == "admin" and pwd == "ros":
                st.session_state.logged_in = True
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")

# --- Dashboard After Login ---
def dashboard():
    st.title("👨‍⚕ Doctor's Dashboard")
    st.markdown("### Welcome back 🩺")

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    if st.sidebar.button("🧠 Scan for 2D"):
        st.session_state.show_2d = True
        st.session_state.show_3d = False
    if st.sidebar.button("🧠 Scan for 3D"):
        st.session_state.show_3d = True
        st.session_state.show_2d = False

    if st.session_state.show_2d:
        scan_for_2d()
    elif st.session_state.show_3d:
        scan_for_3d()
    else:
        st.info("Please select a scan type from sidebar.")

# --- 2D Detection Page ---
def scan_for_2d():
    st.title("Brain Tumor Detection (2D Scan)")
    model = load_model(r'C:\Brain MRI Seg\unet_brain_segmentation.h5')
    IMG_SIZE = (224, 224)
    img_file = st.file_uploader("Upload a .tif Image", type=["tif"], key="i2d")
    mask_file = st.file_uploader("Upload True Mask (.tif)", type=["tif"], key="m2d")
    if img_file:
        img = np.array(Image.open(img_file))
        inp = cv2.resize(img, IMG_SIZE) / 255.0
        pred = model.predict(inp[np.newaxis, ...])[0]
        mask = (pred > 0.5).astype(np.uint8) * 255
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[:, :, 0]
        st.subheader("Original")
        st.image(img, channels="RGB")
        st.subheader("Predicted Mask")
        st.image(mask, clamp=True, channels="GRAY")
        if mask_file:
            true = np.array(Image.open(mask_file))
            true = cv2.resize(true, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
            true_bin = (true > 127).astype(np.uint8) * 255
            acc = (mask.flatten() == true_bin.flatten()).mean() * 100
            st.success(f"Pixel Accuracy: {acc:.2f}%")
        buf = io.BytesIO()
        Image.fromarray(mask).save(buf, format="TIFF")
        st.download_button("Download Mask", buf.getvalue(), "pred_mask.tif", "image/tiff")

# --- 3D Detection Page ---
def scan_for_3d():
    st.title("Brain Tumor Detection (3D Visualization)")
    img_file = st.file_uploader("Upload brain slice (.tif)", type=["tif"], key="i3d")
    mask_file = st.file_uploader("Upload tumor mask (.tif)", type=["tif"], key="m3d")
    slices = st.slider("Number of slices", 10, 30, 20, 5)

    if img_file and mask_file:
        with st.spinner('Creating fast 3D visualization...'):
            try:
                img2d = np.array(Image.open(img_file).convert('L'))
                mask2d = np.array(Image.open(mask_file).convert('L'))

                rows, cols = img2d.shape
                y, x = np.ogrid[:rows, :cols]
                center_x, center_y = cols // 2, rows // 2

                tumor_mask = (mask2d > 127).astype(np.uint8) * 255

                brain_volume = np.zeros((slices, rows, cols), dtype=np.uint8)
                tumor_volume = np.zeros_like(brain_volume)

                z_values = np.linspace(-1, 1, slices)

                for idx, z in enumerate(z_values):
                    r = int(min(center_x, center_y) * np.sqrt(1 - z ** 2))
                    if r <= 0:
                        continue
                    brain_mask = ((x - center_x) ** 2 + (y - center_y) ** 2 <= r ** 2)
                    brain_volume[idx][brain_mask] = img2d[brain_mask]

                    if np.any(tumor_mask):
                        tumor_slice = np.zeros_like(img2d)
                        tumor_slice[brain_mask] = tumor_mask[brain_mask]
                        tumor_volume[idx] = tumor_slice

                fig = go.Figure()

                X, Y, Z = np.mgrid[:brain_volume.shape[1], :brain_volume.shape[2], :brain_volume.shape[0]]

                fig.add_trace(go.Isosurface(
                    x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                    value=brain_volume.transpose(2,1,0).flatten(),
                    isomin=50, isomax=255,
                    opacity=0.4,
                    surface_count=2,
                    colorscale=[[0, 'lightblue'], [1, 'blue']],
                    showscale=False,
                    caps=dict(x_show=False, y_show=False, z_show=False)
                ))

                if np.any(tumor_volume):
                    fig.add_trace(go.Isosurface(
                        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                        value=tumor_volume.transpose(2,1,0).flatten(),
                        isomin=50, isomax=255,
                        opacity=0.9,
                        surface_count=1,
                        colorscale=[[0, 'red'], [1, 'darkred']],
                        showscale=False,
                        caps=dict(x_show=False, y_show=False, z_show=False)
                    ))

                fig.update_layout(
                    scene=dict(
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        zaxis=dict(visible=False),
                        aspectmode='data',
                        camera_eye=dict(x=1.5, y=-1.5, z=0.7)
                    ),
                    margin=dict(l=0, r=0, b=0, t=30),
                    height=600
                )

                st.plotly_chart(fig, use_container_width=True)

                if np.any(tumor_volume):
                    voxels = np.sum(tumor_volume > 0)
                    center = np.mean(np.argwhere(tumor_volume > 0), axis=0)
                    st.markdown(f"""
                        **Tumor Statistics:**
                        - Volume: {voxels} voxels
                        - Center: Z={center[0]:.0f}, Y={center[1]:.0f}, X={center[2]:.0f}
                        - Approx. Diameter: {int(2 * (voxels ** (1/3)))} voxels
                    """)

            except Exception as e:
                st.error(f"Processing error: {str(e)}")

# --- Run App ---
if not st.session_state.logged_in:
    login()
else:
    dashboard()
