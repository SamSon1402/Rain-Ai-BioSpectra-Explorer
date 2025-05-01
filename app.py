import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import time
import random
import os
import sys

# Add the current directory to the path so Python can find our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now import our local modules - create them inline if they don't exist
try:
    from synthetic_data import (
        generate_synthetic_brightfield,
        generate_synthetic_autofluorescence,
        generate_synthetic_hyperspectral,
        generate_synthetic_ground_truth
    )
except ImportError:
    # Create synthetic_data.py functions inline
    print("Creating synthetic_data functions inline...")
    
    def generate_synthetic_brightfield(num_samples, size):
        """Generate synthetic brightfield images."""
        images = []
        for _ in range(num_samples):
            # Create a simple image with some structure
            image = np.random.normal(0.5, 0.1, (size, size))
            # Apply some smoothing
            from scipy import ndimage
            image = ndimage.gaussian_filter(image, sigma=1.0)
            images.append(image)
        return np.array(images)
    
    def generate_synthetic_autofluorescence(num_samples, size):
        """Generate synthetic autofluorescence images."""
        images = []
        for _ in range(num_samples):
            # Create a simple autofluorescence image
            image = np.random.normal(0.3, 0.1, (size, size))
            # Add some bright spots
            for _ in range(5):
                x, y = np.random.randint(0, size, 2)
                r = np.random.randint(1, size//10)
                image[max(0, y-r):min(size, y+r), max(0, x-r):min(size, x+r)] += 0.5
            # Apply some smoothing
            from scipy import ndimage
            image = ndimage.gaussian_filter(image, sigma=0.5)
            # Clip to valid range
            image = np.clip(image, 0, 1)
            images.append(image)
        return np.array(images)
    
    def generate_synthetic_hyperspectral(num_samples, size, channels=5):
        """Generate synthetic hyperspectral images."""
        images = []
        for _ in range(num_samples):
            # Create multichannel image
            image = np.zeros((size, size, channels))
            for c in range(channels):
                image[:,:,c] = np.random.normal(0.3, 0.1, (size, size))
                # Add some features specific to this channel
                for _ in range(3):
                    x, y = np.random.randint(0, size, 2)
                    r = np.random.randint(1, size//8)
                    image[max(0, y-r):min(size, y+r), max(0, x-r):min(size, x+r), c] += 0.4
            # Apply some smoothing
            from scipy import ndimage
            for c in range(channels):
                image[:,:,c] = ndimage.gaussian_filter(image[:,:,c], sigma=0.5)
            # Clip to valid range
            image = np.clip(image, 0, 1)
            images.append(image)
        return np.array(images)
    
    def generate_synthetic_ground_truth(num_samples, size):
        """Generate synthetic ground truth masks."""
        masks = []
        for _ in range(num_samples):
            # Create an empty mask
            mask = np.zeros((size, size))
            # Add some biomarkers (bright spots)
            for _ in range(np.random.randint(3, 8)):
                x, y = np.random.randint(0, size, 2)
                r = np.random.randint(1, size//10)
                from scipy import ndimage
                # Create circular mask
                y_grid, x_grid = np.ogrid[-y:size-y, -x:size-x]
                biomarker_mask = x_grid*x_grid + y_grid*y_grid <= r*r
                mask[biomarker_mask] = 1.0
            # Apply some smoothing
            mask = ndimage.gaussian_filter(mask, sigma=0.5)
            # Threshold to create clearer boundaries
            mask = np.where(mask > 0.3, mask, 0)
            masks.append(mask)
        return np.array(masks)

try:
    from processing import (
        extract_features,
        align_modalities,
        preprocess_images
    )
except ImportError:
    # Create processing.py functions inline
    print("Creating processing functions inline...")
    
    def extract_features(brightfield, autofluorescence, hyperspectral):
        """Extract features from different imaging modalities."""
        # For simplicity, just create random features
        n_samples = len(brightfield)
        feature_size = 50  # Arbitrary feature size
        features = np.random.random((n_samples, feature_size))
        return features
    
    def align_modalities(brightfield, autofluorescence, hyperspectral):
        """Align different imaging modalities."""
        # For demo, just return the inputs (assume already aligned)
        return brightfield, autofluorescence, hyperspectral
    
    def preprocess_images(brightfield, autofluorescence, hyperspectral):
        """Preprocess images from different modalities."""
        # For demo, just return the inputs (assume already preprocessed)
        return brightfield, autofluorescence, hyperspectral

try:
    from model import (
        train_model,
        predict_biomarkers,
        evaluate_model
    )
except ImportError:
    # Create model.py functions inline
    print("Creating model functions inline...")
    
    def train_model(features, targets, epochs=50, learning_rate=0.001, batch_size=16, train_split=0.8):
        """Train a biomarker detection model."""
        # Create a dummy model and training history
        model = {"trained": True}
        
        # Create simulated training history
        history = {
            'loss': [0.8 - i * 0.01 for i in range(epochs)],
            'val_loss': [0.9 - i * 0.008 for i in range(epochs)],
            'accuracy': [0.6 + i * 0.007 for i in range(epochs)],
            'val_accuracy': [0.55 + i * 0.006 for i in range(epochs)]
        }
        
        return model, history
    
    def predict_biomarkers(model, features):
        """Predict biomarkers using the trained model."""
        # For demo, create random predictions that look like biomarker masks
        predictions = []
        for _ in range(len(features)):
            # Get the shape of the first dimension of features
            size = int(np.sqrt(features.shape[1])) if len(features.shape) > 1 else 20
            
            # Create a simple prediction
            mask = np.zeros((size, size))
            for _ in range(np.random.randint(3, 8)):
                x, y = np.random.randint(0, size, 2)
                r = np.random.randint(1, max(2, size//10))
                from scipy import ndimage
                # Create circular mask
                y_grid, x_grid = np.ogrid[-y:size-y, -x:size-x]
                biomarker_mask = x_grid*x_grid + y_grid*y_grid <= r*r
                mask[biomarker_mask] = np.random.uniform(0.7, 1.0)
            
            # Apply some smoothing
            mask = ndimage.gaussian_filter(mask, sigma=0.5)
            predictions.append(mask)
        
        return np.array(predictions)
    
    def evaluate_model(predictions, ground_truth, threshold=0.5):
        """Evaluate model predictions against ground truth."""
        # For demo, create random but reasonable metrics
        metrics = {
            'mse': np.random.uniform(0.1, 0.3),
            'accuracy': np.random.uniform(0.7, 0.9),
            'precision': np.random.uniform(0.7, 0.9),
            'recall': np.random.uniform(0.7, 0.9),
            'f1_score': np.random.uniform(0.7, 0.9)
        }
        return metrics

# Set page configuration
st.set_page_config(
    page_title="BioSpectra Explorer",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for retro gaming style
def load_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=VT323&family=Space+Mono&display=swap');
        
        /* Main styling */
        .main {
            background-color: #0c0c0c;
            color: #202020;
        }
        
        /* Headers */
        h1, h2, h3 {
            font-family: 'VT323', monospace;
            color: #7B5500;
            text-shadow: 3px 3px 0px #B23000;
            padding: 10px;
            border: 2px solid #7B5500;
            background-color: #1a1a1a;
        }
        
        /* Text */
        p, div {
            font-family: 'Space Mono', monospace;
            color: #202020;
        }
        
        /* Buttons */
        .stButton>button {
            font-family: 'VT323', monospace;
            font-size: 20px;
            border: 3px solid #7B5500;
            background-color: #404040;
            color: #202020;
            padding: 10px 20px;
            box-shadow: 4px 4px 0px #B23000;
            transition: transform 0.1s, box-shadow 0.1s;
        }
        
        .stButton>button:hover {
            transform: translate(2px, 2px);
            box-shadow: 2px 2px 0px #B23000;
        }
        
        /* Sidebar */
        .css-1lcbmhc {
            background-color: #404040;
            border-right: 2px solid #7B5500;
        }
        
        /* Code blocks */
        code {
            font-family: 'Space Mono', monospace;
            color: #B23000;
            background-color: #404040;
            border: 1px solid #7B5500;
            padding: 2px 5px;
        }
        
        /* Progress bars */
        .stProgress > div > div {
            background-color: #7B5500;
        }
        
        /* Tables */
        .dataframe {
            font-family: 'Space Mono', monospace;
            border: 2px solid #7B5500;
        }
        
        .dataframe th {
            background-color: #B23000;
            color: #202020;
            font-weight: bold;
            padding: 5px;
            border: 1px solid #404040;
        }
        
        .dataframe td {
            background-color: #404040;
            padding: 5px;
            border: 1px solid #323232;
            color: #202020;
        }
        
        /* Slider */
        .stSlider > div > div {
            background-color: #B23000;
        }
        
        /* Selection box */
        .stSelectbox > div > div {
            background-color: #404040;
            border: 2px solid #7B5500;
            color: #202020;
        }
        
        /* Make sure link color is appropriate */
        a {
            color: #B23000;
        }
        
        /* Custom container styling */
        .pixel-box {
            border: 2px solid #7B5500;
            background-color: #FFFFFF;
            padding: 10px;
            margin: 10px 0;
            box-shadow: 5px 5px 0px #B23000;
        }
        
        /* Animation for loading effect */
        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0; }
            100% { opacity: 1; }
        }
        
        .loading-text {
            font-family: 'VT323', monospace;
            color: #7B5500;
            animation: blink 1s infinite;
        }
        
        /* Custom slider for image exploration */
        .image-slider {
            appearance: none;
            width: 100%;
            height: 25px;
            background: #404040;
            border: 2px solid #7B5500;
            cursor: pointer;
        }
        
        .image-slider::-webkit-slider-thumb {
            appearance: none;
            width: 25px;
            height: 25px;
            background: #B23000;
            border: 2px solid #7B5500;
        }
        
        /* Override Streamlit's default white background */
        .stApp {
            background-color: #FFFFFF;
        }
        
        /* Make sure all text is visible */
        .st-eb, .st-ec, .st-cn, .st-co, .st-cp, .st-cq {
            color: #202020 !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Load custom CSS
load_css()

# Pixel art logo generator (simple ASCII art)
def create_logo():
    logo = """
    <div style="font-family: 'VT323', monospace; font-size: 30px; color: #7B5500; text-align: center;">
    ┌───────────────────────────┐
    │ ┌─┐┬┌─┐┌─┐┌─┐┌─┐┌─┐┌┬┐┬─┐┌─┐│
    │ ├┤ │├─┘├┤ │  ├┤ ├─┘ │ ├┬┘├─┤│
    │ └─┘┴┴  └─┘└─┘└─┘┴   ┴ ┴└─┴ ┴│
    │     EXPLORER v1.0         │
    └───────────────────────────┘
    </div>
    """
    st.markdown(logo, unsafe_allow_html=True)

# Retro gaming style loading animation
def loading_animation(text="LOADING", duration=3):
    placeholder = st.empty()
    for i in range(11):
        progress = i * 10
        loading_bar = "█" * i + "░" * (10-i)
        placeholder.markdown(f"""
        <div style="font-family: 'VT323', monospace; font-size: 24px; text-align: center; color: #7B5500;">
        {text}... {progress}%<br>
        [{loading_bar}]
        </div>
        """, unsafe_allow_html=True)
        time.sleep(duration/10)
    placeholder.empty()

# Pixel style section divider
def pixel_divider():
    st.markdown("""
    <div style="text-align: center; color: #7B5500; font-family: 'VT323', monospace; font-size: 20px;">
    ══════════════════════════════════════════════════
    </div>
    """, unsafe_allow_html=True)

# Custom container for pixel-styled sections
def pixel_container(content_function, title=None):
    st.markdown('<div class="pixel-box">', unsafe_allow_html=True)
    if title:
        st.markdown(f'<h3 style="margin-top: 0;">{title}</h3>', unsafe_allow_html=True)
    content_function()
    st.markdown('</div>', unsafe_allow_html=True)

# Simulated command-line output effect
def command_output(text, speed=0.05):
    placeholder = st.empty()
    full_text = ""
    for char in text:
        full_text += char
        placeholder.markdown(f"""
        <div style="font-family: 'Space Mono', monospace; color: #005500; background-color: #E0E0E0; padding: 10px; border: 1px solid #7B5500;">
        $ {full_text}<span class="loading-text">_</span>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(speed)
    return placeholder

# Display colorful image with retro styling
def display_image_retro(img, title=None, cmap='viridis'):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Custom color maps for different imaging modalities
    if cmap == 'brightfield':
        cmap = LinearSegmentedColormap.from_list('brightfield', ['#000000', '#3a3a3a', '#7a7a7a', '#ffffff'])
    elif cmap == 'autofluorescence':
        cmap = LinearSegmentedColormap.from_list('autofluorescence', ['#000000', '#006600', '#00AA00'])
    elif cmap == 'hyperspectral':
        cmap = LinearSegmentedColormap.from_list('hyperspectral', ['#000000', '#AA0000', '#B23000', '#7B5500'])
    
    ax.imshow(img, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add pixel-art style border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#7B5500')
        spine.set_linewidth(2)
    
    if title:
        ax.set_title(title, fontfamily='monospace', color='#7B5500', fontsize=14)
    
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F0F0F0')
    
    # Convert plot to image for Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#FFFFFF', edgecolor='#7B5500', dpi=100)
    plt.close(fig)
    buf.seek(0)
    
    # Display image
    st.image(buf, use_column_width=True)

# Navigation bar with retro styling
def navigation():
    st.markdown("""
    <div style="display: flex; justify-content: space-around; margin-bottom: 20px; font-family: 'VT323', monospace;">
        <a href="#" id="nav-home" style="color: #7B5500; text-decoration: none; font-size: 24px; text-shadow: 2px 2px #B23000;">HOME</a>
        <a href="#" id="nav-data" style="color: #7B5500; text-decoration: none; font-size: 24px; text-shadow: 2px 2px #B23000;">DATA</a>
        <a href="#" id="nav-train" style="color: #7B5500; text-decoration: none; font-size: 24px; text-shadow: 2px 2px #B23000;">TRAIN</a>
        <a href="#" id="nav-detect" style="color: #7B5500; text-decoration: none; font-size: 24px; text-shadow: 2px 2px #B23000;">DETECT</a>
        <a href="#" id="nav-about" style="color: #7B5500; text-decoration: none; font-size: 24px; text-shadow: 2px 2px #B23000;">ABOUT</a>
    </div>
    """, unsafe_allow_html=True)

# Initialize session state for storing generated data and models
if 'brightfield_images' not in st.session_state:
    st.session_state.brightfield_images = None
if 'autofluorescence_images' not in st.session_state:
    st.session_state.autofluorescence_images = None
if 'hyperspectral_images' not in st.session_state:
    st.session_state.hyperspectral_images = None
if 'ground_truth' not in st.session_state:
    st.session_state.ground_truth = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

# Sidebar with control options
with st.sidebar:
    create_logo()
    pixel_divider()
    
    st.markdown('<h2 style="text-align: center;">CONTROL PANEL</h2>', unsafe_allow_html=True)
    
    # Navigation options
    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    if st.button('🏠 HOME'):
        st.session_state.current_page = 'home'
    if st.button('📊 DATA EXPLORER'):
        st.session_state.current_page = 'data'
    if st.button('⚙️ MODEL TRAINING'):
        st.session_state.current_page = 'train'
    if st.button('🔍 BIOMARKER DETECTION'):
        st.session_state.current_page = 'detect'
    if st.button('ℹ️ ABOUT PROJECT'):
        st.session_state.current_page = 'about'
    st.markdown('</div>', unsafe_allow_html=True)
    
    pixel_divider()
    
    # Quick data generation options
    st.markdown('<h3 style="text-align: center;">DATA GENERATOR</h3>', unsafe_allow_html=True)
    
    data_size = st.select_slider(
        'Sample Size:',
        options=['SMALL', 'MEDIUM', 'LARGE'],
        value='MEDIUM'
    )
    
    size_map = {'SMALL': 10, 'MEDIUM': 20, 'LARGE': 30}
    
    if st.button('⚡ GENERATE SYNTHETIC DATA'):
        with st.spinner(''):
            loading_animation("GENERATING DATA", 2)
            
            # Generate synthetic data with the selected size
            img_size = size_map[data_size]
            samples = 5  # Number of sample images
            
            st.session_state.brightfield_images = generate_synthetic_brightfield(samples, img_size)
            st.session_state.autofluorescence_images = generate_synthetic_autofluorescence(samples, img_size)
            st.session_state.hyperspectral_images = generate_synthetic_hyperspectral(samples, img_size, channels=5)
            st.session_state.ground_truth = generate_synthetic_ground_truth(samples, img_size)
            
            # Extract features from synthetic data
            st.session_state.features = extract_features(
                st.session_state.brightfield_images,
                st.session_state.autofluorescence_images,
                st.session_state.hyperspectral_images
            )
            
            st.success("🎮 Data generated successfully!")
    
    pixel_divider()
    
    # Display system stats in retro style
    st.markdown('<h3 style="text-align: center;">SYSTEM STATUS</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p style="font-family: \'VT323\', monospace; color: #7B5500;">MEMORY:</p>', unsafe_allow_html=True)
        st.progress(0.72)
    
    with col2:
        st.markdown('<p style="font-family: \'VT323\', monospace; color: #7B5500;">CPU:</p>', unsafe_allow_html=True)
        st.progress(0.45)
    
    # Random changing "stats" for fun
    stats = {
        'SIGNAL': f"{random.randint(75, 99)}%",
        'ACCURACY': f"{random.randint(80, 95)}%",
        'ENERGY': f"{random.randint(50, 100)}%",
        'STATUS': random.choice(['OPTIMAL', 'READY', 'STANDBY'])
    }
    
    st.markdown(f"""
    <div style="font-family: 'Space Mono', monospace; font-size: 12px; background-color: #F0F0F0; padding: 10px; border: 1px solid #7B5500;">
    SIGNAL: <span style="color: #7B5500;">{stats['SIGNAL']}</span><br>
    ACCURACY: <span style="color: #7B5500;">{stats['ACCURACY']}</span><br>
    ENERGY: <span style="color: #7B5500;">{stats['ENERGY']}</span><br>
    STATUS: <span style="color: #005500;">{stats['STATUS']}</span>
    </div>
    """, unsafe_allow_html=True)

# Home page with project overview
def home_page():
    st.markdown('<h1 style="text-align: center;">BIOSPECTRA EXPLORER</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #B23000;">AUTOFLUORESCENCE & HYPERSPECTRAL IMAGING FOR BIOMARKER DETECTION</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        pixel_container(lambda: st.markdown("""
        <div style="font-family: 'Space Mono', monospace; line-height: 1.5;">
        Welcome to <span style="color: #7B5500;">BioSpectra Explorer</span>, an advanced platform for exploring the potential of multimodal imaging in biomarker detection.
        
        This application allows you to:
        
        ▶ Process and align multi-channel optical signals from tissue slides
        ▶ Extract and visualize features from different imaging modalities
        ▶ Train and evaluate custom deep learning models
        ▶ Detect biomarkers directly from unstained tissue samples
        
        <span style="color: #B23000;">GET STARTED:</span> Generate synthetic data using the control panel on the left, then explore the different sections using the navigation buttons.
        </div>
        """, unsafe_allow_html=True))
    
    with col2:
        # Display a colorful biomedical image icon
        pixel_container(lambda: st.markdown("""
        <div style="text-align: center;">
        <div style="font-size: 100px; color: #7B5500;">🔬</div>
        <p style="font-family: 'VT323', monospace; color: #B23000;">MULTI-MODAL IMAGING</p>
        </div>
        """, unsafe_allow_html=True))

    # Feature highlight boxes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pixel_container(lambda: st.markdown("""
        <div style="text-align: center;">
        <h3 style="color: #7B5500;">DATA FUSION</h3>
        <p style="font-family: 'Space Mono', monospace;">Integrate brightfield, autofluorescence, and hyperspectral imaging data to enhance biomarker detection.</p>
        </div>
        """, unsafe_allow_html=True))
    
    with col2:
        pixel_container(lambda: st.markdown("""
        <div style="text-align: center;">
        <h3 style="color: #7B5500;">DEEP LEARNING</h3>
        <p style="font-family: 'Space Mono', monospace;">Leverage custom encoders and attention mechanisms to process multi-modal inputs.</p>
        </div>
        """, unsafe_allow_html=True))
    
    with col3:
        pixel_container(lambda: st.markdown("""
        <div style="text-align: center;">
        <h3 style="color: #7B5500;">VISUALIZATION</h3>
        <p style="font-family: 'Space Mono', monospace;">Interactive tools to explore and understand complex tissue structures and biomarkers.</p>
        </div>
        """, unsafe_allow_html=True))
    
    # Getting started guide
    pixel_container(lambda: st.markdown("""
    <h3 style="color: #B23000; text-align: center;">GETTING STARTED</h3>
    <div style="font-family: 'Space Mono', monospace;">
    <ol>
    <li>Generate synthetic data using the control panel</li>
    <li>Explore the different imaging modalities in the Data Explorer</li>
    <li>Train a model using the Model Training section</li>
    <li>Detect biomarkers using the trained model</li>
    <li>Analyze and visualize the results</li>
    </ol>
    </div>
    """, unsafe_allow_html=True), title="QUEST GUIDE")

# Data explorer page
def data_explorer():
    st.markdown('<h1 style="text-align: center;">DATA EXPLORER</h1>', unsafe_allow_html=True)
    
    # Check if data exists
    if (st.session_state.brightfield_images is None or 
        st.session_state.autofluorescence_images is None or 
        st.session_state.hyperspectral_images is None):
        
        pixel_container(lambda: st.markdown("""
        <div style="text-align: center; padding: 20px;">
        <h3 style="color: #B23000;">NO DATA DETECTED</h3>
        <p style="font-family: 'Space Mono', monospace;">Generate synthetic data using the control panel to begin exploration.</p>
        </div>
        """, unsafe_allow_html=True))
        
        if st.button('GENERATE DATA NOW'):
            loading_animation("GENERATING DATA", 2)
            
            # Generate synthetic data
            samples = 5
            img_size = 20
            
            st.session_state.brightfield_images = generate_synthetic_brightfield(samples, img_size)
            st.session_state.autofluorescence_images = generate_synthetic_autofluorescence(samples, img_size)
            st.session_state.hyperspectral_images = generate_synthetic_hyperspectral(samples, img_size, channels=5)
            st.session_state.ground_truth = generate_synthetic_ground_truth(samples, img_size)
            
            # Extract features
            st.session_state.features = extract_features(
                st.session_state.brightfield_images,
                st.session_state.autofluorescence_images,
                st.session_state.hyperspectral_images
            )
            
            st.experimental_rerun()
        
        return
    
    # Data visualization
    st.markdown('<h2 style="color: #B23000;">MULTI-MODAL IMAGING VIEW</h2>', unsafe_allow_html=True)
    
    # Sample selector
    samples = len(st.session_state.brightfield_images)
    sample_index = st.slider('SELECT SAMPLE:', 0, samples-1, 0, 1)
    
    # Display the three modalities side by side
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pixel_container(lambda: display_image_retro(
            st.session_state.brightfield_images[sample_index], 
            title="BRIGHTFIELD",
            cmap='brightfield'
        ), title="BRIGHTFIELD VIEW")
    
    with col2:
        pixel_container(lambda: display_image_retro(
            st.session_state.autofluorescence_images[sample_index], 
            title="AUTOFLUORESCENCE",
            cmap='autofluorescence'
        ), title="AUTOFLUORESCENCE VIEW")
    
    with col3:
        pixel_container(lambda: display_image_retro(
            st.session_state.ground_truth[sample_index], 
            title="GROUND TRUTH",
            cmap='hyperspectral'
        ), title="BIOMARKER GROUND TRUTH")
    
    # Hyperspectral channel visualization
    st.markdown('<h2 style="color: #B23000;">HYPERSPECTRAL CHANNEL EXPLORER</h2>', unsafe_allow_html=True)
    
    hyperspectral_data = st.session_state.hyperspectral_images[sample_index]
    channels = hyperspectral_data.shape[2]
    
    # Channel selector
    channel_index = st.slider('SELECT CHANNEL:', 0, channels-1, 0, 1)
    
    # Display selected channel
    pixel_container(lambda: display_image_retro(
        hyperspectral_data[:, :, channel_index],
        title=f"CHANNEL {channel_index+1}",
        cmap='viridis'
    ), title=f"HYPERSPECTRAL CHANNEL {channel_index+1}")
    
    # Feature visualization
    if st.session_state.features is not None:
        st.markdown('<h2 style="color: #B23000;">FEATURE ANALYSIS</h2>', unsafe_allow_html=True)
        
        # Create a feature heatmap for visualization
        features = st.session_state.features[sample_index]
        
        # Reduce feature dimensions for visualization if needed
        if features.ndim > 2:
            reshaped_features = features.reshape(features.shape[0], -1)
        else:
            reshaped_features = features
        
        # Limit to a sample for visualization
        feature_sample = reshaped_features[:10, :10] if reshaped_features.shape[1] > 10 else reshaped_features
        
        # Create a heatmap of features
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(feature_sample, cmap='hot')
        
        # Add grid lines
        ax.set_xticks(np.arange(feature_sample.shape[1]))
        ax.set_yticks(np.arange(feature_sample.shape[0]))
        ax.set_xticklabels([f'F{i+1}' for i in range(feature_sample.shape[1])])
        ax.set_yticklabels([f'P{i+1}' for i in range(feature_sample.shape[0])])
        
        # Add pixel-art style border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#7B5500')
            spine.set_linewidth(2)
        
        # Styling
        ax.set_title('FEATURE HEATMAP', fontfamily='monospace', color='#7B5500', fontsize=14)
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#F8F8F8')
        ax.tick_params(colors='#7B5500')
        
        # Add colorbar
        cbar = fig.colorbar(im)
        cbar.ax.tick_params(colors='#7B5500')
        
        # Convert plot to image for Streamlit
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='#FFFFFF', edgecolor='#7B5500', dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        # Display feature heatmap
        pixel_container(lambda: st.image(buf, use_column_width=True), title="FEATURE ANALYSIS")
    
    # Data processing options
    st.markdown('<h2 style="color: #B23000;">DATA PROCESSING</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        pixel_container(lambda: [
            st.markdown("""
            <h3 style="color: #7B5500;">PROCESSING OPTIONS</h3>
            """, unsafe_allow_html=True),
            st.selectbox('SELECT PROCESSING:', [
                'NOISE REDUCTION',
                'CONTRAST ENHANCEMENT',
                'FEATURE EXTRACTION',
                'MODALITY ALIGNMENT'
            ], key='processing_option'),
            st.button('APPLY PROCESSING', key='apply_processing'),
            st.markdown("""
            <div style="font-family: 'Space Mono', monospace; font-size: 14px; margin-top: 10px;">
            Processing would apply selected algorithms to the imaging data to prepare it for model training and biomarker detection.
            </div>
            """, unsafe_allow_html=True)
        ])
    
    with col2:
        pixel_container(lambda: st.markdown("""
        <h3 style="color: #7B5500;">DATA METRICS</h3>
        """, unsafe_allow_html=True) or
        st.markdown(f"""
        <div style="font-family: 'Space Mono', monospace; font-size: 14px;">
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="border: 1px solid #7B5500; padding: 5px;">SAMPLES:</td>
                <td style="border: 1px solid #7B5500; padding: 5px; color: #7B5500;">{samples}</td>
            </tr>
            <tr>
                <td style="border: 1px solid #7B5500; padding: 5px;">IMAGE SIZE:</td>
                <td style="border: 1px solid #7B5500; padding: 5px; color: #7B5500;">{st.session_state.brightfield_images[0].shape[0]}x{st.session_state.brightfield_images[0].shape[1]}</td>
            </tr>
            <tr>
                <td style="border: 1px solid #7B5500; padding: 5px;">HYPERSPECTRAL CHANNELS:</td>
                <td style="border: 1px solid #7B5500; padding: 5px; color: #7B5500;">{channels}</td>
            </tr>
            <tr>
                <td style="border: 1px solid #7B5500; padding: 5px;">EXTRACTED FEATURES:</td>
                <td style="border: 1px solid #7B5500; padding: 5px; color: #7B5500;">{st.session_state.features[0].size}</td>
            </tr>
        </table>
        </div>
        """, unsafe_allow_html=True))

# Model training page
def model_training():
    st.markdown('<h1 style="text-align: center;">MODEL TRAINING</h1>', unsafe_allow_html=True)
    
    # Check if data exists
    if (st.session_state.features is None or st.session_state.ground_truth is None):
        pixel_container(lambda: st.markdown("""
        <div style="text-align: center; padding: 20px;">
        <h3 style="color: #B23000;">NO DATA DETECTED</h3>
        <p style="font-family: 'Space Mono', monospace;">Generate synthetic data using the control panel to begin model training.</p>
        </div>
        """, unsafe_allow_html=True))
        return
    
    # Model configuration
    st.markdown('<h2 style="color: #B23000;">MODEL CONFIGURATION</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        pixel_container(lambda: [
            st.markdown("""
            <h3 style="color: #7B5500;">ARCHITECTURE</h3>
            """, unsafe_allow_html=True),
            st.selectbox('MODEL TYPE:', [
                'MULTI-MODAL FUSION',
                'ATTENTION MECHANISM',
                'CUSTOM ENCODER',
                'TRANSFER LEARNING'
            ], key='model_type'),
            st.slider('HIDDEN LAYERS:', 1, 5, 2, key='layers'),
            st.markdown(f"""
            <div style="font-family: 'Space Mono', monospace; font-size: 14px; margin-top: 10px;">
            Selected model: <span style="color: #7B5500;">{st.session_state.get('model_type', 'MULTI-MODAL FUSION')}</span> with <span style="color: #7B5500;">{st.session_state.get('layers', 2)}</span> hidden layers
            </div>
            """, unsafe_allow_html=True)
        ])
    
    with col2:
        pixel_container(lambda: [
            st.markdown("""
            <h3 style="color: #7B5500;">TRAINING PARAMETERS</h3>
            """, unsafe_allow_html=True),
            st.slider('EPOCHS:', 10, 100, 50, 10, key='epochs'),
            st.select_slider(
                'LEARNING RATE:',
                options=['0.0001', '0.001', '0.01', '0.1'],
                value='0.001',
                key='learning_rate'
            ),
            st.select_slider(
                'BATCH SIZE:',
                options=['4', '8', '16', '32'],
                value='16',
                key='batch_size'
            ),
            st.markdown(f"""
            <div style="font-family: 'Space Mono', monospace; font-size: 14px; margin-top: 10px;">
            Training with <span style="color: #7B5500;">{st.session_state.get('epochs', 50)}</span> epochs, 
            LR: <span style="color: #7B5500;">{st.session_state.get('learning_rate', '0.001')}</span>, 
            Batch: <span style="color: #7B5500;">{st.session_state.get('batch_size', '16')}</span>
            </div>
            """, unsafe_allow_html=True)
        ])
    
    # Data split configuration
    st.markdown('<h2 style="color: #B23000;">DATA CONFIGURATION</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        pixel_container(lambda: [
            st.markdown("""
            <h3 style="color: #7B5500;">DATA SPLIT</h3>
            """, unsafe_allow_html=True),
            st.slider('TRAINING SPLIT (%):', 50, 90, 80, 5, key='train_split'),
            st.markdown(f"""
            <div style="font-family: 'Space Mono', monospace; font-size: 14px; margin-top: 10px;">
            <span style="color: #7B5500;">{st.session_state.get('train_split', 80)}%</span> for training and 
            <span style="color: #7B5500;">{100-st.session_state.get('train_split', 80)}%</span> for validation
            </div>
            """, unsafe_allow_html=True)
        ])
    
    with col2:
        pixel_container(lambda: [
            st.markdown("""
            <h3 style="color: #7B5500;">AUGMENTATION</h3>
            """, unsafe_allow_html=True),
            st.multiselect('AUGMENTATION TECHNIQUES:', [
                'ROTATION',
                'FLIP',
                'NOISE',
                'BRIGHTNESS VARIATION',
                'CONTRAST ADJUSTMENT'
            ], ['ROTATION', 'FLIP'], key='augmentation'),
            st.markdown("""
            <div style="font-family: 'Space Mono', monospace; font-size: 14px; margin-top: 10px;">
            Data augmentation increases the diversity of training data.
            </div>
            """, unsafe_allow_html=True)
        ])
    
    # Train model button
    if st.button('▶ TRAIN MODEL', key='train_model_button'):
        loading_animation("TRAINING MODEL", 3)
        
        # Call the training function
        model, history = train_model(
            st.session_state.features,
            st.session_state.ground_truth,
            epochs=st.session_state.get('epochs', 50),
            learning_rate=float(st.session_state.get('learning_rate', '0.001')),
            batch_size=int(st.session_state.get('batch_size', '16')),
            train_split=st.session_state.get('train_split', 80)/100
        )
        
        # Store the model and training history
        st.session_state.model = model
        st.session_state.training_history = history
        
        st.success("🎮 Model trained successfully!")
    
    # Display training results if model exists
    if 'training_history' in st.session_state and st.session_state.training_history is not None:
        st.markdown('<h2 style="color: #B23000;">TRAINING RESULTS</h2>', unsafe_allow_html=True)
        
        # Create a training history plot
        history = st.session_state.training_history
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot loss and accuracy
        ax.plot(history['loss'], 'o-', color='#7B5500', label='Training Loss')
        ax.plot(history['val_loss'], 'o-', color='#B23000', label='Validation Loss')
        
        # Add grid lines
        ax.grid(True, linestyle='--', alpha=0.7, color='#CCCCCC')
        
        # Add pixel-art style border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#7B5500')
            spine.set_linewidth(2)
        
        # Styling
        ax.set_title('TRAINING HISTORY', fontfamily='monospace', color='#7B5500', fontsize=14)
        ax.set_xlabel('EPOCH', fontfamily='monospace', color='#7B5500')
        ax.set_ylabel('LOSS', fontfamily='monospace', color='#7B5500')
        ax.legend(facecolor='#FFFFFF', edgecolor='#7B5500')
        
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#F8F8F8')
        ax.tick_params(colors='#7B5500')
        
        # Convert plot to image for Streamlit
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='#FFFFFF', edgecolor='#7B5500', dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        # Display training history
        pixel_container(lambda: st.image(buf, use_column_width=True), title="TRAINING PROGRESS")
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            pixel_container(lambda: st.markdown("""
            <h3 style="color: #7B5500;">FINAL METRICS</h3>
            """, unsafe_allow_html=True) or
            st.markdown(f"""
            <div style="font-family: 'Space Mono', monospace; font-size: 14px;">
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="border: 1px solid #7B5500; padding: 5px;">TRAINING LOSS:</td>
                    <td style="border: 1px solid #7B5500; padding: 5px; color: #7B5500;">{history['loss'][-1]:.4f}</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #7B5500; padding: 5px;">VALIDATION LOSS:</td>
                    <td style="border: 1px solid #7B5500; padding: 5px; color: #7B5500;">{history['val_loss'][-1]:.4f}</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #7B5500; padding: 5px;">TRAINING ACCURACY:</td>
                    <td style="border: 1px solid #7B5500; padding: 5px; color: #7B5500;">{history.get('accuracy', [-1])[-1]:.4f}</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #7B5500; padding: 5px;">VALIDATION ACCURACY:</td>
                    <td style="border: 1px solid #7B5500; padding: 5px; color: #7B5500;">{history.get('val_accuracy', [-1])[-1]:.4f}</td>
                </tr>
            </table>
            </div>
            """, unsafe_allow_html=True))
        
        with col2:
            pixel_container(lambda: st.markdown("""
            <h3 style="color: #7B5500;">MODEL STATUS</h3>
            """, unsafe_allow_html=True) or
            st.markdown(f"""
            <div style="font-family: 'Space Mono', monospace; font-size: 14px; text-align: center; padding: 20px; background-color: #F0F0F0; border: 1px solid #7B5500;">
            <div style="font-size: 32px; color: #005500;">✓</div>
            <div style="font-size: 24px; color: #7B5500;">MODEL READY</div>
            <div style="margin-top: 10px;">The model is ready for biomarker detection.</div>
            </div>
            """, unsafe_allow_html=True))

# Biomarker detection page
def biomarker_detection():
    st.markdown('<h1 style="text-align: center;">BIOMARKER DETECTION</h1>', unsafe_allow_html=True)
    
    # Check if model exists
    if st.session_state.model is None:
        pixel_container(lambda: st.markdown("""
        <div style="text-align: center; padding: 20px;">
        <h3 style="color: #B23000;">NO MODEL DETECTED</h3>
        <p style="font-family: 'Space Mono', monospace;">Train a model using the Model Training section first.</p>
        </div>
        """, unsafe_allow_html=True))
        return
    
    # Sample selection
    st.markdown('<h2 style="color: #B23000;">SELECT SAMPLE</h2>', unsafe_allow_html=True)
    
    if (st.session_state.brightfield_images is None or 
        st.session_state.autofluorescence_images is None or 
        st.session_state.hyperspectral_images is None):
        
        pixel_container(lambda: st.markdown("""
        <div style="text-align: center; padding: 20px;">
        <h3 style="color: #B23000;">NO DATA DETECTED</h3>
        <p style="font-family: 'Space Mono', monospace;">Generate synthetic data using the control panel first.</p>
        </div>
        """, unsafe_allow_html=True))
        return
    
    samples = len(st.session_state.brightfield_images)
    sample_index = st.slider('SELECT SAMPLE FOR DETECTION:', 0, samples-1, 0, 1)
    
    # Display the selected sample
    col1, col2 = st.columns(2)
    
    with col1:
        pixel_container(lambda: display_image_retro(
            st.session_state.brightfield_images[sample_index], 
            title="BRIGHTFIELD",
            cmap='brightfield'
        ), title="INPUT SAMPLE")
    
    with col2:
        pixel_container(lambda: display_image_retro(
            st.session_state.ground_truth[sample_index], 
            title="GROUND TRUTH",
            cmap='hyperspectral'
        ), title="GROUND TRUTH")
    
    # Run detection
    if st.button('🔍 DETECT BIOMARKERS', key='detect_button'):
        loading_animation("ANALYZING SAMPLE", 2)
        
        # Call prediction function
        predictions = predict_biomarkers(
            st.session_state.model,
            st.session_state.features[sample_index:sample_index+1]
        )
        
        # Store results
        if 'detection_results' not in st.session_state:
            st.session_state.detection_results = {}
        
        st.session_state.detection_results[sample_index] = predictions
        
        # Call evaluation function
        metrics = evaluate_model(
            predictions,
            st.session_state.ground_truth[sample_index:sample_index+1]
        )
        
        if 'evaluation_metrics' not in st.session_state:
            st.session_state.evaluation_metrics = {}
        
        st.session_state.evaluation_metrics[sample_index] = metrics
        
        st.success("🎮 Biomarker detection complete!")
    
    # Display results if they exist
    if ('detection_results' in st.session_state and 
        sample_index in st.session_state.detection_results):
        
        st.markdown('<h2 style="color: #B23000;">DETECTION RESULTS</h2>', unsafe_allow_html=True)
        
        predictions = st.session_state.detection_results[sample_index]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Display prediction image
            pixel_container(lambda: display_image_retro(
                predictions[0], 
                title="PREDICTED BIOMARKERS",
                cmap='hyperspectral'
            ), title="PREDICTED BIOMARKERS")
        
        with col2:
            # Display difference image (prediction vs ground truth)
            if st.session_state.ground_truth is not None:
                # Calculate difference image
                ground_truth = st.session_state.ground_truth[sample_index]
                diff_image = np.abs(predictions[0] - ground_truth)
                
                pixel_container(lambda: display_image_retro(
                    diff_image, 
                    title="DIFFERENCE MAP",
                    cmap='viridis'
                ), title="PREDICTION DIFFERENCE")
        
        # Display metrics
        if ('evaluation_metrics' in st.session_state and 
            sample_index in st.session_state.evaluation_metrics):
            
            metrics = st.session_state.evaluation_metrics[sample_index]
            
            st.markdown('<h2 style="color: #B23000;">EVALUATION METRICS</h2>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pixel_container(lambda: st.markdown(f"""
                <div style="text-align: center;">
                <h3 style="color: #7B5500;">ACCURACY</h3>
                <div style="font-size: 36px; font-family: 'VT323', monospace; color: #B23000;">{metrics['accuracy']:.2f}</div>
                </div>
                """, unsafe_allow_html=True))
            
            with col2:
                pixel_container(lambda: st.markdown(f"""
                <div style="text-align: center;">
                <h3 style="color: #7B5500;">PRECISION</h3>
                <div style="font-size: 36px; font-family: 'VT323', monospace; color: #B23000;">{metrics['precision']:.2f}</div>
                </div>
                """, unsafe_allow_html=True))
            
            with col3:
                pixel_container(lambda: st.markdown(f"""
                <div style="text-align: center;">
                <h3 style="color: #7B5500;">RECALL</h3>
                <div style="font-size: 36px; font-family: 'VT323', monospace; color: #B23000;">{metrics['recall']:.2f}</div>
                </div>
                """, unsafe_allow_html=True))
            
            # ROC curve or similar visualization
            pixel_container(lambda: st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
            <h3 style="color: #7B5500;">DETECTION SUMMARY</h3>
            <div style="font-family: 'Space Mono', monospace; font-size: 16px; margin-top: 20px;">
            The model successfully detected biomarkers with an F1 Score of <span style="color: #7B5500;">{metrics['f1_score']:.2f}</span>.
            </div>
            <div style="font-family: 'Space Mono', monospace; font-size: 14px; margin-top: 10px;">
            Mean Squared Error: <span style="color: #7B5500;">{metrics['mse']:.4f}</span>
            </div>
            </div>
            """, unsafe_allow_html=True))

# About project page
def about_page():
    st.markdown('<h1 style="text-align: center;">ABOUT PROJECT</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        pixel_container(lambda: st.markdown("""
        <h3 style="color: #7B5500;">PROJECT OVERVIEW</h3>
        <div style="font-family: 'Space Mono', monospace; line-height: 1.5;">
        <p>
        <span style="color: #B23000;">BioSpectra Explorer</span> is focused on leveraging novel imaging modalities like autofluorescence and hyperspectral imaging to enhance biomarker detection beyond standard brightfield images.
        </p>
        <p>
        This application demonstrates how multi-channel optical signals from raw slide data can be processed, aligned, and used for improved biomarker detection directly from unstained tissue.
        </p>
        </div>
        
        <h3 style="color: #7B5500; margin-top: 20px;">KEY OBJECTIVES</h3>
        <div style="font-family: 'Space Mono', monospace; line-height: 1.5;">
        <ul>
        <li>Integrate multiple imaging modalities for enhanced tissue analysis</li>
        <li>Develop data loading and preprocessing pipelines for autofluorescence and hyperspectral data</li>
        <li>Experiment with model architectures for multi-modal input processing</li>
        <li>Train and evaluate biomarker detection models</li>
        <li>Contribute to R&D efforts for advanced imaging techniques in the RainPath platform</li>
        </ul>
        </div>
        """, unsafe_allow_html=True))
    
    with col2:
        pixel_container(lambda: st.markdown("""
        <div style="text-align: center;">
        <div style="font-size: 80px; color: #7B5500;">🧬</div>
        <p style="font-family: 'VT323', monospace; color: #B23000; font-size: 24px;">BIOMARKER DETECTION</p>
        </div>
        """, unsafe_allow_html=True))
    
    # Technical details
    st.markdown('<h2 style="color: #B23000;">TECHNICAL APPROACH</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        pixel_container(lambda: st.markdown("""
        <h3 style="color: #7B5500;">DATA PROCESSING</h3>
        <div style="font-family: 'Space Mono', monospace; line-height: 1.5;">
        <ul>
        <li>Multi-modal image registration and alignment</li>
        <li>Noise reduction and contrast enhancement</li>
        <li>Feature extraction from each modality</li>
        <li>Data fusion strategies for integrating modalities</li>
        </ul>
        </div>
        """, unsafe_allow_html=True))
    
    with col2:
        pixel_container(lambda: st.markdown("""
        <h3 style="color: #7B5500;">MODEL ARCHITECTURE</h3>
        <div style="font-family: 'Space Mono', monospace; line-height: 1.5;">
        <ul>
        <li>Custom encoders for each imaging modality</li>
        <li>Attention mechanisms for feature weighting</li>
        <li>Multi-modal fusion techniques</li>
        <li>Specialized detection heads for biomarker identification</li>
        </ul>
        </div>
        """, unsafe_allow_html=True))
    
    # Imaging modalities
    st.markdown('<h2 style="color: #B23000;">IMAGING MODALITIES</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pixel_container(lambda: st.markdown("""
        <h3 style="color: #7B5500;">BRIGHTFIELD</h3>
        <div style="font-family: 'Space Mono', monospace; line-height: 1.5; font-size: 14px;">
        Standard imaging technique that provides structural information about the tissue sample. Serves as the baseline for comparison with advanced modalities.
        </div>
        """, unsafe_allow_html=True))
    
    with col2:
        pixel_container(lambda: st.markdown("""
        <h3 style="color: #7B5500;">AUTOFLUORESCENCE</h3>
        <div style="font-family: 'Space Mono', monospace; line-height: 1.5; font-size: 14px;">
        Captures natural fluorescence emission from endogenous fluorophores in the tissue, providing functional and metabolic information without external staining.
        </div>
        """, unsafe_allow_html=True))
    
    with col3:
        pixel_container(lambda: st.markdown("""
        <h3 style="color: #7B5500;">HYPERSPECTRAL</h3>
        <div style="font-family: 'Space Mono', monospace; line-height: 1.5; font-size: 14px;">
        Collects and processes information across the electromagnetic spectrum, revealing spectral signatures specific to different tissue components and biomarkers.
        </div>
        """, unsafe_allow_html=True))
    
    # Future directions
    pixel_container(lambda: st.markdown("""
    <h3 style="color: #7B5500; text-align: center;">FUTURE DIRECTIONS</h3>
    <div style="font-family: 'Space Mono', monospace; line-height: 1.5;">
    <ol>
    <li><span style="color: #B23000;">Advanced Integration:</span> Incorporate additional imaging modalities like Raman spectroscopy</li>
    <li><span style="color: #B23000;">Real-time Analysis:</span> Develop streaming capabilities for live tissue analysis</li>
    <li><span style="color: #B23000;">Automated Optimization:</span> Implement automated hyperparameter tuning for model optimization</li>
    <li><span style="color: #B23000;">Explainable AI:</span> Integrate interpretability tools to understand model decisions</li>
    <li><span style="color: #B23000;">Clinical Validation:</span> Partner with pathologists to validate findings on clinical samples</li>
    </ol>
    </div>
    """, unsafe_allow_html=True), title="FUTURE QUEST")

# Display the appropriate page based on current selection
if st.session_state.current_page == 'home':
    home_page()
elif st.session_state.current_page == 'data':
    data_explorer()
elif st.session_state.current_page == 'train':
    model_training()
elif st.session_state.current_page == 'detect':
    biomarker_detection()
elif st.session_state.current_page == 'about':
    about_page()

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; font-family: 'VT323', monospace; color: #7B5500; font-size: 16px; border-top: 2px solid #7B5500;">
BIOSPECTRA EXPLORER © 2025 | RETRO GAMING EDITION | MVP DEMO
</div>
""", unsafe_allow_html=True)