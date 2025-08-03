import base64
import streamlit as st

# --- Page Configuration ---
st.set_page_config(page_title="Doctor Dashboard | MRI Tumor Detection", page_icon="🧠", layout="centered")

# --- Session State Initialization ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# --- Custom CSS Styling ---
def set_background():
    with open(r"C:\Users\Valeska\OneDrive\Desktop\Sem6MiniProject\MRISegmentation\bg.png", "rb") as f:
        data = f.read()
        encoded_image = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            font-family: 'Arial', sans-serif;
            }}
            .title {{
                text-align: center;
                font-size: 40px;
                font-weight: bold;
                color: #ffff;
            }}
            .subtitle {{
                text-align: center;
                font-size: 22px;
                color: #ffff;
            }}
            .stButton>button {{
                background-color: #3498db;
                color: white;
                height: 50px;
                width: 100%;
                border-radius: 8px;
                font-size: 18px;
            }}
            .stTextInput>div>div>input {{
                border: 1px solid #3498db;
                padding: 10px;
                border-radius: 8px;
            }}
            </style>
        """, unsafe_allow_html=True)

# --- Login Form ---
def login():
    set_background()

    st.markdown("<div class='title'>🧠 NEURONIX</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>MRI Brain Tumor Detection System</div>", unsafe_allow_html=True)

    # st.markdown("#### 👨‍⚕️ Doctor Login Portal")

    # Hide sidebar during login
    hide_sidebar_style = """
        <style>
        [data-testid="stSidebar"] {display: none;}
        </style>
    """
    st.markdown(hide_sidebar_style, unsafe_allow_html=True)

    with st.form("login_form"):
        st.write(" Doctor Login Portal")
        username = st.text_input("👨‍⚕️ Doctor ID", placeholder="Enter your Doctor ID")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your Password")
        login_btn = st.form_submit_button("Login")

        if login_btn:
            if username == "valeska" and password == "asdf":
                st.session_state.logged_in = True
                st.success("✅ Login successful! Redirecting to your dashboard...")
                st.experimental_rerun()
            else:
                st.error("❌ Invalid Doctor ID or Password.")

# --- Dashboard After Login ---
def dashboard():
    st.title("👨‍⚕️ Doctor's Dashboard")
    st.markdown("### Welcome back, **Dr. Valeska!** 🩺")

    st.markdown("### 🧑‍💼 Patient Information")
    st.info("""
    - **Name:** John Doe  
    - **Age:** 45  
    - **Scan Date:** April 20, 2025  
    - **Symptoms:** Headache, nausea, blurred vision  
    """)

# --- Run App ---
if not st.session_state.logged_in:
    login()
else:
    dashboard()
