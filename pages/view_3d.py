import streamlit as st
import nibabel as nib
import numpy as np
import tempfile
import os
from skimage import measure
import plotly.graph_objects as go
import numpy as np
import os
import nibabel as nib
import cv2
import matplotlib.pyplot as plt

def save_to_file(file_name, content):
    with open(file_name, "w") as file:
        file.write(content)
        
def load_nifti(uploaded_file):
    """Load NIfTI file with better error handling"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii') as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        
        img = nib.load(tmp_path)
        return img.get_fdata(), img.affine, img.header
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None, None
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

def generate_3d_plot(mri_data=None, seg_data=None):
    """Generate 3D plot that works with either MRI or segmentation alone"""
    fig = go.Figure()
    
    # If MRI data provided, create brain surface
    if mri_data is not None:
        mri_data = mri_data[::2, ::2, ::2]  # Downsample
        try:
            verts, faces, _, _ = measure.marching_cubes(mri_data, level=0.3)
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color='lightblue',
                opacity=0.2,
                name='Brain Anatomy'
            ))
        except Exception as e:
            st.warning(f"Couldn't extract brain surface: {str(e)}")
    
    # If segmentation provided, create tumor surface
    if seg_data is not None:
        seg_data = seg_data[::2, ::2, ::2]  # Downsample
        try:
            verts_t, faces_t, _, _ = measure.marching_cubes(seg_data, level=0.5)
            fig.add_trace(go.Mesh3d(
                x=verts_t[:, 0], y=verts_t[:, 1], z=verts_t[:, 2],
                i=faces_t[:, 0], j=faces_t[:, 1], k=faces_t[:, 2],
                color='red',
                opacity=0.9,
                name='Tumor'
            ))
        except Exception as e:
            st.error(f"Couldn't extract tumor surface: {str(e)}")
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=600
    )
    return fig

def calculate_volumes(seg_data, header):
    """Calculate tumor subregion volumes"""
    voxel_vol = np.prod(header.get_zooms())  # mm³ per voxel
    return {
        'Total Tumor': np.sum(seg_data > 0) * voxel_vol,
        'Necrotic Core': np.sum(seg_data == 1) * voxel_vol,
        'Edema': np.sum(seg_data == 2) * voxel_vol,
        'Enhancing Tumor': np.sum(seg_data == 4) * voxel_vol
    }

def get_tumor_location(seg_data, affine):
    """Calculate tumor center of mass with axis labels"""
    coords = np.argwhere(seg_data > 0)
    com_voxel = coords.mean(axis=0)
    com_mm = nib.affines.apply_affine(affine, com_voxel)
    
    lobe = "Right" if com_mm[0] > 0 else "Left"
    
    # Format coordinates with axis labels
    labeled_coords = [
        f"X (Left-Right): {com_mm[0]:.1f} mm",
        f"Y (Posterior-Anterior): {com_mm[1]:.1f} mm", 
        f"Z (Inferior-Superior): {com_mm[2]:.1f} mm"
    ]
    
    return {
        'Center of Mass': labeled_coords,
        'Estimated Hemisphere': lobe
    }

def analyze_shape(seg_data):
    """Calculate shape characteristics with error handling"""
    binary_mask = (seg_data > 0).astype(int)
    
    try:
        verts, faces, _, _ = measure.marching_cubes(binary_mask, level=0.5)
        surface_area = measure.mesh_surface_area(verts, faces)
        volume = np.sum(binary_mask)
        
        # Handle division by zero cases
        if surface_area > 0 and volume > 0:
            compactness = (36 * np.pi * volume**2) / (surface_area**3)
            sphericity = (compactness * (36*np.pi))**(1/3)
        else:
            compactness = float('nan')
            sphericity = float('nan')
            
        return {
            'Surface Area (mm²)': surface_area,
            'Compactness': compactness,
            'Sphericity': sphericity
        }
    except Exception as e:
        st.warning(f"Shape analysis failed: {str(e)}")
        return {
            'Surface Area (mm²)': float('nan'),
            'Compactness': float('nan'),
            'Sphericity': float('nan')
        }
def generate_treatment_plan(volumes, location, shape):
    """Generate simplified personalized treatment recommendations"""
    tumor_vol = volumes['Total Tumor']
    enh_ratio = volumes['Enhancing Tumor'] / volumes['Total Tumor'] if volumes['Total Tumor'] > 0 else 0
    sphericity = shape['Sphericity']
    
    recs = []
    
    # 1. Surgical recommendation based on size
    if tumor_vol < 3000:  # ~3 cm³
        recs.append("✅ **Good surgical candidate** - Complete resection likely")
    elif tumor_vol < 10000:  # ~10 cm³
        recs.append("⚠️ **Moderate surgical case** - May require staged procedures")
    else:
        recs.append("❌ **Complex surgical case** - Biopsy consideration first")
    
    # 2. Key location consideration
    if "Left" in location['Estimated Hemisphere']:
        recs.append("🗣️ **Left hemisphere tumor** - Language area precautions needed")
    
    # 3. Tumor biology indication
    if enh_ratio > 0.7:
        recs.append("🔬 **Aggressive features** - Likely needs combined therapy")
    elif enh_ratio > 0.3:
        recs.append("📊 **Intermediate features** - May respond to targeted therapy")
    
    # 4. Shape consideration
    if not np.isnan(sphericity):
        if sphericity > 0.7:
            recs.append("⚪ **Well-defined borders** - Clear surgical margins expected")
        else:
            recs.append("🔘 **Infiltrative growth** - Close follow-up recommended")
    
    # Determine urgency
    urgency = "URGENT" if (enh_ratio > 0.7 or tumor_vol > 8000) else "Priority" if tumor_vol > 3000 else "Elective"
    
    return recs, urgency

# Streamlit UI
st.set_page_config(page_title="Advanced Tumor Analysis", layout="wide")

# Sidebar for file upload
with st.sidebar:
    st.header("Data Upload")
    mri_file = st.file_uploader("MRI Scan (Optional)", type=['nii', 'nii.gz'])
    seg_file = st.file_uploader("Segmentation Mask (Required)", type=['nii', 'nii.gz'])

# Main content
# ... [keep all your existing imports and functions] ...

# Main content
st.title("Brain Tumor Dashboard")
VOLUME_SLICES = 100 
VOLUME_START_AT = 22

SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC/CORE', # or NON-ENHANCING tumor CORE
    2 : 'EDEMA',
    3 : 'ENHANCING' # original 4 -> converted into 3 later
}
if seg_file:
    # Load data
    seg_data, affine, header = load_nifti(seg_file)
    mri_data, _, _ = load_nifti(mri_file) if mri_file else (None, None, None)
    
    if seg_data is not None:
        # Calculate all metrics
        volumes = calculate_volumes(seg_data, header)
        location = get_tumor_location(seg_data, affine)
        shape = analyze_shape(seg_data)
        
        # Create two columns
        col_viz, col_analysis = st.columns([2, 1])
        
        with col_viz:
            # 3D Visualization
            with st.spinner("Generating 3D visualization..."):
                fig_3d = generate_3d_plot(mri_data, seg_data)
                st.plotly_chart(fig_3d, use_container_width=True)
            
            # 2D Visualization Section
            slice_idx = st.slider("Select Slice", 0, seg_data.shape[2]-1, seg_data.shape[2]//2)
            
            # Create figure with 2 rows and 3 columns
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # First row - Original, Ground Truth, All Classes
            if mri_data is not None:
                _ = axes[0, 0].imshow(mri_data[:, :, slice_idx], cmap='gray')
            axes[0, 0].set_title("Original MRI")
            axes[0, 0].axis('off')
            
            seg_slice = seg_data[:, :, slice_idx]
            _ = axes[0, 1].imshow(seg_slice, vmin=0, vmax=4, cmap='viridis')
            axes[0, 1].set_title("Ground Truth")
            axes[0, 1].axis('off')
            
            _ = axes[0, 2].imshow(seg_slice, vmin=0, vmax=4, cmap='viridis')
            axes[0, 2].set_title("All Classes Predicted")
            axes[0, 2].axis('off')
            
            # Second row - Individual class predictions
            # Necrotic/Core
            core_mask = np.ma.masked_where(seg_slice != 1, seg_slice)
            if mri_data is not None:
                _ = axes[1, 0].imshow(mri_data[:, :, slice_idx], cmap='gray')
            _ = axes[1, 0].imshow(core_mask, cmap='Reds', alpha=0.7, vmin=0, vmax=4)
            axes[1, 0].set_title("NECROTIC/CORE")
            axes[1, 0].axis('off')
            
            # Edema
            edema_mask = np.ma.masked_where(seg_slice != 2, seg_slice)
            if mri_data is not None:
                _ = axes[1, 1].imshow(mri_data[:, :, slice_idx], cmap='gray')
            _ = axes[1, 1].imshow(edema_mask, cmap='Blues', alpha=0.7, vmin=0, vmax=4)
            axes[1, 1].set_title("EDEMA")
            axes[1, 1].axis('off')
            
            # Enhancing
            enh_mask = np.ma.masked_where(seg_slice != 4, seg_slice)
            if mri_data is not None:
                _ = axes[1, 2].imshow(mri_data[:, :, slice_idx], cmap='gray')
            _ = axes[1, 2].imshow(enh_mask, cmap='Greens', alpha=0.7, vmin=0, vmax=4)
            axes[1, 2].set_title("ENHANCING")
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)


        with col_analysis:
            # [Keep all your existing analysis code unchanged]
            st.header("Tumor Details")
            
            # Volumetric Analysis
            with st.expander("Volumetric Analysis", expanded=True):
                for k, v in volumes.items():
                    st.metric(label=k, value=f"{v:.2f} mm³")
            
            # Spatial Analysis
            with st.expander("Location Details"):
                for coord in location['Center of Mass']:
                    st.write(coord)
                
                st.metric(
                    label="Estimated Hemisphere",
                    value=location['Estimated Hemisphere'],
                    help="Left hemisphere typically contains language centers"
                )
             # Morphological Analysis
            with st.expander("Morphological Analysis"):
                st.metric("Surface Area", f"{shape['Surface Area (mm²)']:.2f} mm²")
                
                if not np.isnan(shape['Sphericity']):
                    st.metric("Sphericity", f"{shape['Sphericity']:.3f}")
                    if shape['Sphericity'] > 0.7:
                        st.success("Highly spherical tumor")
                    elif shape['Sphericity'] > 0.5:
                        st.info("Moderately spherical tumor")
                    else:
                        st.warning("Irregular tumor shape")
                else:
                    st.warning("Could not calculate sphericity")
            
            # Personalized Treatment Plan
            with st.expander("Clinical Summary", expanded=True):
                recommendations, urgency = generate_treatment_plan(volumes, location, shape)
                
                for rec in recommendations:
                    st.markdown(f"- {rec}")
                
                # st.divider()
                # st.markdown(f"**Clinical Priority**: {urgency} case")
            inference = st.text_input("Doctor's notes", placeholder="Points by doctor")
            file_name_txt = st.text_input("File Name for Labels", value="labels.txt")


            if st.button("Save Text"):
                if inference:
                    save_to_file(file_name_txt, inference)
                    st.success(f"Text file '{file_name_txt}' saved successfully!")
                else:
                    st.warning("Please enter some inference details before saving.")
            
            # [Rest of your existing analysis code...]

    
    if seg_data is not None:
        # Calculate all metrics
        volumes = calculate_volumes(seg_data, header)
        location = get_tumor_location(seg_data, affine)
        shape = analyze_shape(seg_data)
        
        # Create two columns
        col_viz, col_analysis = st.columns([2, 1])
        

       
           
    else:
        st.error("Failed to load segmentation data")
else:
    st.warning("Please upload at least a segmentation mask to begin analysis")

# Add some styling
st.markdown("""
<style>
[data-testid="stMetricValue"] {
    font-size: 1.1rem;
}
[data-testid="stMetricLabel"] {
    font-size: 0.9rem;
}
.st-emotion-cache-16idsys p {
    font-family: monospace;
}
[data-testid="stExpander"] {
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)
