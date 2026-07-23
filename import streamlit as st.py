import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import requests
from streamlit_image_coordinates import streamlit_image_coordinates
from skimage.color import rgb2lab

# ==========================================
# 1. PAGE SETUP
# ==========================================
st.set_page_config(page_title="Color Finder Pro", layout="wide")
st.title("🎨 Color Finder Pro")
st.write("Click anywhere on your image to instantly identify the color name, LAB values, and match accuracy.")

# ==========================================
# 2. HELPER: CONVERT RGB TO LAB (WHOLE NUMBERS)
# ==========================================
def rgb_to_lab(r, g, b):
    """Converts 0-255 RGB values to integer CIELAB coordinates."""
    rgb_normalized = np.array([[[r / 255.0, g / 255.0, b / 255.0]]], dtype=np.float32)
    lab = rgb2lab(rgb_normalized)[0, 0]
    return int(round(lab[0])), int(round(lab[1])), int(round(lab[2]))

# ==========================================
# 3. LOAD COLOR DATASET & COMPUTE LAB
# ==========================================
CSV_URL = "https://raw.githubusercontent.com/codebrainz/color-names/master/output/colors.csv"
CSV_FILE = "colors.csv"

@st.cache_data
def load_dataset():
    column_names = ["color_name", "hex", "R", "G", "B"]
    
    if not os.path.exists(CSV_FILE):
        try:
            response = requests.get(CSV_URL, timeout=10)
            with open(CSV_FILE, "wb") as f:
                f.write(response.content)
        except Exception:
            # Emergency fallback if internet drops
            return pd.DataFrame([
                ["Pure Black", "#000000", 0, 0, 0, 0, 0, 0], 
                ["Pure White", "#ffffff", 255, 255, 255, 100, 0, 0]
            ], columns=column_names + ["L", "A", "B"])
            
    df = pd.read_csv(CSV_FILE, names=column_names, header=None)
    
    # Pre-calculate integer LAB coordinates for dataset
    rgb_array = df[['R', 'G', 'B']].values.reshape(-1, 1, 3) / 255.0
    lab_array = rgb2lab(rgb_array).reshape(-1, 3)
    
    df['L'] = np.round(lab_array[:, 0]).astype(int)
    df['A'] = np.round(lab_array[:, 1]).astype(int)
    df['B'] = np.round(lab_array[:, 2]).astype(int)
    
    return df

color_df = load_dataset()

# ==========================================
# 4. MATCHING ENGINE (RETURNS MATCH & ACCURACY)
# ==========================================
def find_nearest_shade_lab(target_l, target_a, target_b):
    """Calculates Delta E distance and accuracy percentage."""
    # Euclidean distance in CIELAB space (Delta E)
    distances = np.sqrt(
        (color_df['L'] - target_l) ** 2 + 
        (color_df['A'] - target_a) ** 2 + 
        (color_df['B'] - target_b) ** 2
    )
    
    match_idx = distances.idxmin()
    delta_e = distances[match_idx]
    
    # Calculate accuracy percentage (100% minus Delta E error)
    accuracy = max(0.0, 100.0 - delta_e)
    
    return color_df.loc[match_idx], round(accuracy, 1)

# ==========================================
# 5. SIDEBAR INPUTS
# ==========================================
mode = st.sidebar.radio("Image Source:", ("📤 Upload Picture", "📷 Use Webcam"))
target_image = None

if mode == "📤 Upload Picture":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        target_image = Image.open(uploaded_file).convert("RGB")
else:
    webcam_file = st.camera_input("Take a snapshot")
    if webcam_file is not None:
        target_image = Image.open(webcam_file).convert("RGB")

# ==========================================
# 6. SIDE-BY-SIDE INTERACTIVE LAYOUT
# ==========================================
if target_image is not None:
    col_left, col_right = st.columns([1.2, 1.0], gap="large")
    
    with col_left:
        st.write("### 🎯 Step 1: Click a color below")
        target_image.thumbnail((600, 600))
        img_width, img_height = target_image.size
        
        click_data = streamlit_image_coordinates(target_image, width=img_width, key=f"picker_{mode}")
    
    with col_right:
        st.write("### 📊 Step 2: View Results")
        if click_data is not None:
            x, y = click_data["x"], click_data["y"]
            
            if x < img_width and y < img_height:
                # Extract RGB of clicked pixel
                img_array = np.array(target_image)
                r, g, b = img_array[y, x]
                
                # Convert clicked pixel to integer CIELAB
                l_val, a_val, b_val = rgb_to_lab(r, g, b)
                
                # Find closest match & calculate match accuracy
                matched_row, accuracy_score = find_nearest_shade_lab(l_val, a_val, b_val)
                color_name = matched_row['color_name']
                
                # --- LAB & ACCURACY RESULT CARD ---
                st.markdown(f"""
                <div style="background-color: #ffffff; padding: 25px; border-radius: 12px; border: 1px solid #e0e0e0; box-shadow: 0px 4px 12px rgba(0,0,0,0.05);">
                    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 12px;">
                        <div style="background-color: rgb({r}, {g}, {b}); width: 55px; height: 55px; border-radius: 50%; border: 2px solid #ccc; flex-shrink: 0;"></div>
                        <div>
                            <h2 style="margin: 0; color: #111; font-size: 1.8em;">{color_name}</h2>
                            <p style="margin: 4px 0 0 0; color: #555; font-size: 1.1em;">
                                <b>LAB Value:</b> <code>L*: {l_val}, a*: {a_val}, b*: {b_val}</code>
                            </p>
                        </div>
                    </div>
                    <hr style="border: 0; border-top: 1px solid #eee; margin: 12px 0;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <span style="color: #444; font-weight: bold; font-size: 1.05em;">Match Accuracy:</span>
                        <span style="background-color: #e6f4ea; color: #137333; font-weight: bold; padding: 4px 12px; border-radius: 16px; font-size: 1.1em;">
                            {accuracy_score}%
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Click anywhere on the image to view its color name, LAB values, and match accuracy.")
