import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io
from streamlit_image_comparison import image_comparison
import warnings 
warnings.filterwarnings("ignore")

def negative_transformation(image):
    return cv2.bitwise_not(image)

def log_transformation(image):
    c = 255 / (np.log(1 + np.max(image)))
    log_image = c * (np.log(1 + image.astype(np.float32)))
    return np.clip(log_image, 0, 255).astype(np.uint8)

def bitplane_slicing(image, bitplane):
    return (image & (1 << bitplane)) * 255

def gray_level_slicing(image, min_range, max_range):
    sliced = np.where((image >= min_range) & (image <= max_range), 255, 0)
    return sliced.astype(np.uint8)

def contrast_stretching(image):
    min_val, max_val = np.min(image), np.max(image)
    stretched = (image - min_val) * (255 / (max_val - min_val))
    return stretched.astype(np.uint8)

def histogram_stretching(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

def apply_mean_filter(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))

def apply_median_filter(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)

def apply_gaussian_filter(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def laplacian_filter(image):
    laplacian=cv2.Laplacian(image, cv2.CV_64F)
    return cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def sobel_filter(image, axis="x"):
    """
    Apply Sobel filter with improved normalization
    """
    if axis == "x":
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    elif axis == "y":
        sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    else:
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize the image to [0, 255]
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return sobel

# def laplacian_filter(image):
#     return cv2.Laplacian(image, cv2.CV_64F)

def prewitt_filter(image, axis="x"):
    if axis == "x":
        kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    elif axis == "y":
        kernel = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    else:
        return image
    return cv2.filter2D(image, -1, kernel)

def canny_edge_detection(image, threshold1, threshold2):
    return cv2.Canny(image, threshold1, threshold2)

def adjust_brightness(image, alpha, beta):
    """Adjust brightness and contrast of an image."""
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def gamma_correction(image, gamma):
    """Apply gamma correction."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def histogram_equalization(image):
    """Apply Histogram Equalization."""
    if len(image.shape) == 3 and image.shape[2] == 3:  # For color images
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:  # For grayscale images
        enhanced = cv2.equalizeHist(image)
    return enhanced

def low_pass_filter(image, d):
    """Apply a corrected Low Pass Filter using the ideal frequency filter."""
    rows, cols = image.shape[:2]
    crow, ccol = rows // 2, cols // 2  # center of the image
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), d, 1, -1)  # Create a circular mask
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    filtered = dft_shift * mask[:, :, np.newaxis]  # Apply the mask
    dft_ishift = np.fft.ifftshift(filtered)
    img_back = cv2.idft(dft_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])  # Compute magnitude
    return np.clip(img_back / np.max(img_back) * 255, 0, 255).astype(np.uint8)

def high_pass_filter(image, d):
    """Apply a corrected High Pass Filter using the ideal frequency filter."""
    rows, cols = image.shape[:2]
    crow, ccol = rows // 2, cols // 2  # center of the image
    mask = np.ones((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), d, 0, -1)  # Create a circular mask (inverted for HPF)
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    filtered = dft_shift * mask[:, :, np.newaxis]  # Apply the mask
    dft_ishift = np.fft.ifftshift(filtered)
    img_back = cv2.idft(dft_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])  # Compute magnitude
    return np.clip(img_back / np.max(img_back) * 255, 0, 255).astype(np.uint8)

def min_filter(image, kernel_size):
    """Apply Minimum Filter."""
    return cv2.erode(image, np.ones((kernel_size, kernel_size), np.uint8))

def max_filter(image, kernel_size):
    """Apply Maximum Filter."""
    return cv2.dilate(image, np.ones((kernel_size, kernel_size), np.uint8))

# Add these functions after imports
def resize_image(image, max_size=800):
    """Resize image while maintaining aspect ratio"""
    height, width = image.shape[:2]
    if height > max_size or width > max_size:
        ratio = max_size / max(height, width)
        new_size = (int(width * ratio), int(height * ratio))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image

def create_expandable_image(image, title):
    st.image(image, use_container_width=True)
    if st.button("🔍", key=f"zoom_{title}"):
        with st.expander("", expanded=True):
            st.image(image, use_container_width=True)
            if st.button("✖️", key=f"close_{title}"):
                st.session_state[f'zoom_{title}'] = False

# Update local_css() with theme compatibility
def local_css():
    st.markdown("""
    <style>
    /* Theme Compatibility */
    [data-testid="stSidebar"] {
        background-color: var(--background-color);
    }
    
    .stApp {
        background: var(--background-color);
        transition: all 0.3s ease;
    }

    [data-theme="dark"] {
        --background-color: #1E1E1E;
        --text-color: #FFFFFF;
        --card-background: #2D2D2D;
    }

    [data-theme="light"] {
        --background-color: #F0F2F6;
        --text-color: #262730;
        --card-background: #FFFFFF;
    }

    /* Enhanced Card Container */
    .card {
        background-color: var(--card-background);
        border-radius: 1rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }

    .card:hover {
        transform: translateY(-2px);
    }

    /* Image Grid */
    .image-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }

    /* Enhanced Headers */
    h1, h2, h3 {
        color: var(--text-color);
        font-weight: 600;
    }
    .grid-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        padding: 1rem;
    }
    
    .image-card {
        background: var(--card-background);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .zoom-btn {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        background: linear-gradient(45deg, #2980b9, #3498db);
        color: white;
        border: none;
        border-radius: 5px;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

def add_sidebar_info():
    # Brief intro and social links at top
    st.sidebar.markdown("""
    # 🎨 Image Enhancement Studio
    
    This Image Processing application was developed by **Pruthak Jani**  using knowledge gained by the **Image Processing** course at L.D. College of Engineering, Ahmedabad, 2024.
    
    [![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?style=flat&logo=github)](https://github.com/pruthakjani5)
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/pruthak-jani/)
    
    ---
    """)
    # Detailed techniques info in expander
    with st.sidebar.expander("🔍 Learn About Techniques"):
        st.markdown("""
        ### Implemented Techniques
        
        #### 1. Pixel-Level Operations
        - **Negative Transformation**: Inverts pixel intensities
        - **Log Transformation**: Enhances low-intensity details
        - **Bit-Plane Slicing**: Visualizes bit contributions
        - **Gray-Level Slicing**: Highlights specific intensity ranges
        - **Contrast & Histogram Stretching**: Improves image contrast
        
        #### 2. Spatial Domain Filtering
        - **Mean Filter**: Smoothing through averaging
        - **Median Filter**: Noise reduction
        - **Gaussian Filter**: Edge-preserving smoothing
        - **Sobel & Prewitt**: Edge detection
        - **Laplacian**: Edge enhancement
        - **Min/Max Filters**: Morphological operations
        
        #### 3. Color & Brightness
        - **Brightness Adjustment**: Linear scaling
        - **Gamma Correction**: Non-linear intensity mapping
        - **Histogram Equalization**: Contrast enhancement
        
        #### 4. Frequency Domain
        - **Low Pass Filter**: Noise reduction
        - **High Pass Filter**: Edge enhancement
        """)
    # Add a disclaimer at the bottom    
    st.sidebar.markdown("""
    ---
    📚 **Disclaimer**: This app is for educational purposes only
        .
    ### 👨‍💻 Developer
    **Pruthak Jani**  
    L.D. College of Engineering  
    📧 **Contact**: Reach out to me on [LinkedIn](https://www.linkedin.com/in/pruthak-jani/)
    """)

def process_image(original_image, grayscale_image, enhancement_option, enhancement_category):
    enhanced_image = original_image.copy()
    enhanced_grayscale = grayscale_image.copy()

    # Pixel-Level Techniques
    if enhancement_category == "Pixel-Level":
        if enhancement_option == "Negative Transformation":
            enhanced_image = negative_transformation(original_image)
            enhanced_grayscale = negative_transformation(grayscale_image)
        elif enhancement_option == "Log Transformation":
            enhanced_image = log_transformation(original_image)
            enhanced_grayscale = log_transformation(grayscale_image)
        elif enhancement_option == "Bit-Plane Slicing":
            bitplane = st.sidebar.slider("Select Bitplane (0-7)", 0, 7, 0)
            enhanced_image = bitplane_slicing(grayscale_image, bitplane)
            enhanced_grayscale = enhanced_image
        elif enhancement_option == "Gray-Level Slicing":
            min_range = st.sidebar.slider("Minimum Gray Level", 0, 255, 50)
            max_range = st.sidebar.slider("Maximum Gray Level", 0, 255, 200)
            enhanced_image = gray_level_slicing(grayscale_image, min_range, max_range)
            enhanced_grayscale = enhanced_image
        elif enhancement_option == "Contrast Stretching":
            enhanced_image = contrast_stretching(grayscale_image)
            enhanced_grayscale = enhanced_image
        elif enhancement_option == "Histogram Stretching":
            enhanced_image = histogram_stretching(grayscale_image)
            enhanced_grayscale = enhanced_image

    # Spatial Filtering Techniques
    elif enhancement_category == "Spatial":
        if enhancement_option == "Mean Filter":
            kernel_size = st.sidebar.slider("Kernel Size", 3, 15, 5, step=2)
            enhanced_image = apply_mean_filter(original_image, kernel_size)
            enhanced_grayscale = apply_mean_filter(grayscale_image, kernel_size)
        elif enhancement_option == "Median Filter":
            kernel_size = st.sidebar.slider("Kernel Size", 3, 15, 5, step=2)
            enhanced_image = apply_median_filter(original_image, kernel_size)
            enhanced_grayscale = apply_median_filter(grayscale_image, kernel_size)
        elif enhancement_option == "Gaussian Filter":
            kernel_size = st.sidebar.slider("Kernel Size", 3, 15, 5, step=2)
            enhanced_image = apply_gaussian_filter(original_image, kernel_size)
            enhanced_grayscale = apply_gaussian_filter(grayscale_image, kernel_size)
        elif enhancement_option == "Sobel Filter":
            sobel_axis = st.sidebar.radio("Sobel Axis", ["x", "y", "magnitude"])
            enhanced_image = sobel_filter(cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY), sobel_axis)
            enhanced_grayscale = enhanced_image
        elif enhancement_option == "Prewitt Filter":
            prewitt_axis = st.sidebar.radio("Prewitt Axis", ["x", "y"])
            enhanced_image = prewitt_filter(grayscale_image, prewitt_axis)
            enhanced_grayscale = enhanced_image
        elif enhancement_option == "Canny Edge Detection":
            threshold1 = st.sidebar.slider("Threshold 1", 50, 200, 100)
            threshold2 = st.sidebar.slider("Threshold 2", 50, 200, 150)
            enhanced_image = canny_edge_detection(grayscale_image, threshold1, threshold2)
            enhanced_grayscale = enhanced_image
        elif enhancement_option == "Min Filter":
            kernel_size = st.sidebar.slider("Kernel Size", 3, 15, 3, step=2)
            enhanced_image = min_filter(original_image, kernel_size)
            enhanced_grayscale = min_filter(grayscale_image, kernel_size)
        elif enhancement_option == "Max Filter":
            kernel_size = st.sidebar.slider("Kernel Size", 3, 15, 3, step=2)
            enhanced_image = max_filter(original_image, kernel_size)
            enhanced_grayscale = max_filter(grayscale_image, kernel_size)
        elif enhancement_option == "Laplacian Filter":
            enhanced_image = laplacian_filter(grayscale_image)
            enhanced_grayscale = enhanced_image

    # Color/Brightness Techniques
    elif enhancement_category == "Color/Brightness":
        if enhancement_option == "Adjust Brightness":
            alpha = st.sidebar.slider("Contrast (alpha)", 0.5, 3.0, 1.0, step=0.1)
            beta = st.sidebar.slider("Brightness (beta)", -100, 100, 0)
            enhanced_image = adjust_brightness(original_image, alpha, beta)
            enhanced_grayscale = adjust_brightness(grayscale_image, alpha, beta)
        elif enhancement_option == "Gamma Correction":
            gamma = st.sidebar.slider("Gamma", 0.1, 3.0, 1.0, step=0.1)
            enhanced_image = gamma_correction(original_image, gamma)
            enhanced_grayscale = gamma_correction(grayscale_image, gamma)
        elif enhancement_option == "Histogram Equalization":
            enhanced_image = histogram_equalization(original_image)
            enhanced_grayscale = histogram_equalization(grayscale_image)
        # Frequency Filtering Techniques
    elif enhancement_category == "Frequency":
        if enhancement_option == "Low Pass Filter":
            d = st.sidebar.slider("Filter Radius", 1, 50, 10)
            enhanced_image = low_pass_filter(grayscale_image, d)
            enhanced_grayscale = enhanced_image
        elif enhancement_option == "High Pass Filter":
            d = st.sidebar.slider("Filter Radius", 1, 50, 10)
            enhanced_image = high_pass_filter(grayscale_image, d)
            enhanced_grayscale = enhanced_image
    return enhanced_image, enhanced_grayscale

def display_image_grid(original_image, grayscale_image, enhanced_image, enhanced_grayscale):
    col1, col2 = st.columns(2)
    
    # First Row - Original Images
    with col1:
        st.markdown("#### Original Color")
        create_expandable_image(
            cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
            "original_color"
        )
    with col2:
        st.markdown("#### Original Grayscale")
        create_expandable_image(grayscale_image, "original_gray")
    
    # Second Row - Enhanced Images
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Enhanced Color")
        create_expandable_image(
            cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB),
            "enhanced_color"
        )
    with col4:
        st.markdown("#### Enhanced Grayscale")
        create_expandable_image(enhanced_grayscale, "enhanced_gray")

def main():
    local_css()
    add_sidebar_info()
    st.markdown("""
    <h1>🌈 ImProc: Ultimate Image Enhancement Studio</h1>
    """, unsafe_allow_html=True)

    # Sidebar Design
    st.sidebar.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <h2 style='color: #2c3e50; font-weight: 700;'>
        🛠️ Enhancement Workshop
        </h2>
        <p style='color: #7f8c8d; font-style: italic;'>
        Transform Your Images with Advanced Techniques
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # File Uploader with Enhanced Design
    st.markdown("""
    <div class='image-container' style='text-align: center;'>
        <h3 style='color: #2c3e50;'>📤 Upload Your Image</h3>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image", 
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Supports various image formats. Max file size: 200MB"
    )
    
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        # Resize images before processing
        image = resize_image(np.array(Image.open(uploaded_file)))
        
        if len(image.shape) == 3:
            original_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        else:
            original_image = image
            grayscale_image = image

        # Enhanced Enhancement Category Selection
        st.sidebar.markdown("### 📊 Processing Domain")
        enhancement_category = st.sidebar.radio(
            "Choose Your Transformation",
            ["Pixel-Level", "Spatial", "Color/Brightness", "Frequency"],
            help="Select the type of image enhancement technique"
        )

        # Updated Enhancement Options
        enhancement_options = {
            "Pixel-Level": [
                "None", "Negative Transformation", "Log Transformation", 
                "Bit-Plane Slicing", "Gray-Level Slicing", 
                "Contrast Stretching", "Histogram Stretching"
            ],
            "Spatial": [
                "None", "Mean Filter", "Median Filter", "Gaussian Filter", 
                "Sobel Filter", "Prewitt Filter", "Canny Edge Detection","Laplacian Filter", "Min Filter", "Max Filter"
            ],
            "Color/Brightness": [
                "None", "Adjust Brightness", "Gamma Correction", 
                "Histogram Equalization"
            ],
            "Frequency": [
                "None", "Low Pass Filter", "High Pass Filter"
            ]
        }
        
        enhancement_option = st.sidebar.selectbox(
            f"{enhancement_category} Techniques",
            enhancement_options[enhancement_category],
            help=f"Select a specific {enhancement_category.lower()} enhancement technique"
        )

        # Processing remains the same as in your original script
        enhanced_image, enhanced_grayscale = process_image(
            original_image, 
            grayscale_image, 
            enhancement_option,
            enhancement_category
        )

        # Display 2x2 grid of images
        display_image_grid(
            original_image,
            grayscale_image,
            enhanced_image,
            enhanced_grayscale
        )

        # Download Section
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📥 Download Images")
        
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            color_bytes = cv2.imencode('.png', original_image)[1].tobytes()
            st.download_button("💾 Original Color", color_bytes, "original_color.png")
            
        with c2:
            gray_bytes = cv2.imencode('.png', grayscale_image)[1].tobytes()
            st.download_button("💾 Original Gray", gray_bytes, "original_gray.png")
            
        with c3:
            enhanced_color_bytes = cv2.imencode('.png', enhanced_image)[1].tobytes()
            st.download_button("💾 Enhanced Color", enhanced_color_bytes, "enhanced_color.png")
            
        with c4:
            enhanced_gray_bytes = cv2.imencode('.png', enhanced_grayscale)[1].tobytes()
            st.download_button("💾 Enhanced Gray", enhanced_gray_bytes, "enhanced_gray.png")
            
        st.markdown('</div>', unsafe_allow_html=True)

        # Results Display
        st.markdown("### 🖼️ Image Comparisons")
        
        st.markdown('<div class="grid-container">', unsafe_allow_html=True)
        
        # Color Images Comparison
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        image_comparison(
            img1=cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
            img2=cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB),
            label1="Original",
            label2="Enhanced"
        )
        if st.button("Zoom 🔍", key="color_zoom"):
            with st.expander("Color Images", expanded=True):
                st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
                st.image(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Grayscale Images Comparison
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        image_comparison(
            img1=grayscale_image,
            img2=enhanced_grayscale,
            label1="Original Gray",
            label2="Enhanced Gray"
        )
        if st.button("Zoom 🔍", key="gray_zoom"):
            with st.expander("Grayscale Images", expanded=True):
                st.image(grayscale_image)
                st.image(enhanced_grayscale)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Add theme toggle in sidebar
def add_theme_toggle():
    theme = st.sidebar.selectbox(
        "🎨 Choose Theme",
        ["Light", "Dark"],
        help="Switch between light and dark theme"
    )
    
    if theme == "Dark":
        st.markdown("""
            <script>
                document.documentElement.setAttribute('data-theme', 'dark');
            </script>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <script>
                document.documentElement.setAttribute('data-theme', 'light');
            </script>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    st.set_page_config(
        page_title="ImProc: Advanced Image Processing",
        page_icon="🎨",
        layout="wide"
    )
    add_theme_toggle()
    main()
