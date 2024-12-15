# 🌟 ImProc: Advanced Image Enhancement Studio

Welcome to **ImProc**, an advanced image enhancement studio designed to unleash the full potential of your images using cutting-edge image processing techniques. Developed with care and creativity by **Pruthak Jani**, this project is a feature-rich Streamlit application that offers a blend of powerful functionalities and user-friendly design.

## 🚀 Key Features

### 🎨 **Image Enhancement Techniques**

#### 1. Pixel-Level Operations
- **Negative Transformation**: Inverts image pixel intensities for creative effects.
- **Log Transformation**: Enhances low-intensity details effectively.
- **Bit-Plane Slicing**: Visualizes the contributions of each bitplane.
- **Gray-Level Slicing**: Highlights specific intensity ranges dynamically.
- **Contrast & Histogram Stretching**: Boosts image contrast for clarity.

#### 2. Spatial Domain Filtering
- **Mean Filter**: Smoothens images by averaging pixel values.
- **Median Filter**: Reduces noise while preserving edges.
- **Gaussian Filter**: Applies edge-preserving smoothing.
- **Sobel Filter**: Detects edges based on gradient magnitude.
- **Prewitt Filter**: Simple edge detection using convolutional kernels.
- **Laplacian Filter**: Enhances edges through second-derivative analysis.
- **Min/Max Filters**: Morphological operations for custom effects.

#### 3. Color & Brightness Adjustments
- **Brightness Adjustment**: Control brightness with fine-tuning options.
- **Gamma Correction**: Non-linear intensity transformation for vivid images.
- **Histogram Equalization**: Improves contrast using histogram redistribution.

#### 4. Frequency Domain Operations
- **Low Pass Filter**: Removes high-frequency noise for smoother visuals.
- **High Pass Filter**: Highlights edges and fine details by suppressing low frequencies.

### 🖼️ **Interactive Features**
- Real-time previews of original and enhanced images.
- Side-by-side image comparisons with zoom and expander functionalities.
- Fully responsive design, optimized for both light and dark themes.

### 📥 **Output Downloads**
Easily download:
- Original and grayscale images.
- Enhanced images with pixel-perfect quality.

## 🛠️ Tech Stack
- **Python 3.9+**: Core language for robust development.
- **Streamlit**: Simplifies building interactive web applications.
- **OpenCV**: Advanced library for image processing.
- **NumPy**: Efficient numerical computations for image matrices.
- **Pillow (PIL)**: Image handling and manipulation library.

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/pruthakjani5/ImProc.git
   cd ImProc
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

4. Open the app in your browser at `http://localhost:8501`.

## 🚀 Usage
Transform your images in three simple steps:
1. **Upload Your Image**: Select an image in formats like JPG, PNG, or BMP.
2. **Select Techniques**: Choose from pixel-level, spatial, color, or frequency-based enhancements.
3. **Download Results**: Save original and processed images with a click.

Experience the live demo here: [🌐 ImProc App](https://improc-app.streamlit.app/)

## 🛠️ **Customizing the Application**
Want to add or tweak functionalities? The code is modular and well-documented, allowing you to:
- Add new enhancement techniques.
- Customize the UI layout and design.
- Extend functionalities with additional processing methods.

## 🌍 **About the Developer**
**Pruthak Jani** is a passionate engineer pursuing **BE AI ML with a minor in Robotics** at **L.D. College of Engineering**, Ahmedabad. This project embodies his learning and expertise gained from the **Image Processing (3155204)** course in Semester 5.

[![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?style=flat&logo=github)](https://github.com/pruthakjani5)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/pruthak-jani/)

## 🤝 **Contributions and Feedback**
Contributions, issues, and feature requests are welcome! Feel free to:
- Fork the repository and submit a pull request.
- Open issues for bugs or enhancements.

## 📜 **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Disclaimer:** This application is for educational purposes only. The developer is not liable for any misuse or potential damage caused by the tool.

---

Thank you for exploring **ImProc**! Empower your images with innovation and creativity. 🌟
