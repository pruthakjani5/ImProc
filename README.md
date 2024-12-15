# 🌟 ImProc: Advanced Image Enhancement Studio

Welcome to **ImProc**, an advanced image enhancement studio designed to unleash the full potential of your images using cutting-edge image processing techniques. Developed with care and creativity by **Pruthak Jani**, this project is a feature-rich Streamlit application that offers a blend of powerful functionalities and user-friendly design.

## 🎯 Project Overview

This project redefines how images are processed and enhanced, offering seamless and interactive ways to transform visuals. Whether you're a researcher, photographer, or enthusiast, ImProc equips you with tools to amplify the beauty and clarity of your images effortlessly.

## 🏆 Key Highlights

- 🖼️ **Real-Time Enhancements**: Instantly view and tweak enhancements.
- 🌈 **Advanced Techniques**: Pixel-level operations, spatial filters, frequency domain transformations, and more.
- ⚙️ **Customizable**: Modular design for easy integration of new features.
- 📥 **Effortless Downloads**: Save your enhanced images directly.

## ✨ Features

### 1. Pixel-Level Operations 🎨

- **Negative Transformation**: Converts each pixel value to its negative (255 - pixel intensity). Useful for reversing black and white tones and enhancing certain features.
- **Log Transformation**: Applies a logarithmic scaling to compress dynamic range and reveal hidden details in low-intensity regions.
- **Bit-Plane Slicing**: Extracts individual bit-planes from pixel values, visualizing how each bit contributes to the overall intensity.
- **Gray-Level Slicing**: Highlights a specific intensity range while suppressing others. Ideal for isolating features within a specified brightness range.
- **Contrast Stretching**: Enhances the image by stretching the range of intensity values to span the full intensity spectrum (0–255).
- **Histogram Stretching**: Normalizes the image's intensity values to improve contrast by redistributing pixel intensities.

### 2. Spatial Domain Filtering 🌌

- **Mean Filter**: Applies averaging over a pixel neighborhood to smooth the image and reduce noise.
- **Median Filter**: Replaces each pixel with the median value of its neighborhood, effectively reducing salt-and-pepper noise.
- **Gaussian Filter**: Uses a Gaussian kernel for edge-preserving smoothing and blurring effects.
- **Sobel Filter**: Computes image gradients to detect edges along the x, y, or combined directions.
- **Prewitt Filter**: Similar to Sobel but uses simpler convolutional kernels for edge detection.
- **Laplacian Filter**: Highlights areas of rapid intensity change using second-order derivatives, enhancing edges.
- **Min Filter**: Applies morphological erosion to extract minimum intensity values in the kernel neighborhood.
- **Max Filter**: Uses morphological dilation to highlight maximum intensity values in the kernel neighborhood.
- **Canny Edge Detection**: Detects edges by applying Gaussian smoothing, intensity gradient calculation, and non-maximum suppression.

### 3. Color & Brightness Adjustments 🌞

- **Brightness Adjustment**: Scales pixel intensity values linearly using contrast (alpha) and brightness (beta) factors.
- **Gamma Correction**: Non-linear adjustment of pixel intensity values for perceptual brightness changes, useful for over- or under-exposed images.
- **Histogram Equalization**: Enhances contrast by redistributing the intensity histogram, either globally or locally for specific image regions.

### 4. Frequency Domain Operations 📡

- **Low Pass Filter**: Removes high-frequency components (noise) while preserving low-frequency content (details like smooth regions).
- **High Pass Filter**: Retains high-frequency content (edges and fine details) while suppressing low-frequency areas (smooth backgrounds).

### 🖼️ Interactive Features

- 🔍 **Side-by-Side Comparisons**: Compare original vs. enhanced images.
- 🖱️ **Zoom & Expand**: Interactive zoom for detailed inspections.
- 🌗 **Light/Dark Mode**: Adaptive design for comfortable viewing.

## 💻 Tech Stack

- **Python 3.9+**: Core programming language.
- **Streamlit**: Framework for intuitive UI design.
- **OpenCV**: Comprehensive image processing library.
- **NumPy**: Fast numerical computations.
- **Pillow (PIL)**: Image handling and transformations.

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

1. **Upload Your Image**: Drag and drop an image in formats like JPG, PNG, or BMP.
2. **Choose Enhancements**: Select from pixel-level operations, spatial filtering, or brightness/color adjustments.
3. **Download Results**: Save the enhanced versions with a single click.

Experience the live demo here: [🌐 ImProc App](https://improc-app.streamlit.app/)

## 🔧 Customization

This project is designed to be extended:
- Add new filters or transformations.
- Integrate additional file format support.
- Customize the UI for specific use cases.

## 🗂️ Project Structure

```
improc/
│
├── 📄 app.py             # Streamlit application entry point
├── 📄 requirements.txt   # Python dependencies
└── 📄 README.md          # Documentation
```

## 🌍 About the Developer

**Pruthak Jani** is a passionate engineer pursuing **BE AI ML with a minor in Robotics** at **L.D. College of Engineering**, Ahmedabad. This project represents his dedication to bridging academic learning with real-world applications, developed as part of the **Image Processing (3155204)** course in Semester 5.

[![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?style=flat&logo=github)](https://github.com/pruthakjani5)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/pruthak-jani/)

## 🤝 Contributions & Feedback

Contributions, suggestions, and feedback are always welcome! Feel free to:
- 🍴 Fork the repository.
- 🌟 Star the project if you find it helpful.
- 🐛 Report issues or submit pull requests.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

💡 **Disclaimer**: This application is for educational purposes only. The developer is not liable for any misuse or unintended outcomes.

---

Thank you for exploring **ImProc**! Empower your images with innovation and creativity. 🌟
