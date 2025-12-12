# 🌲 Intel Image Classification Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Flask-2.0.1-green.svg)](https://flask.palletsprojects.com/)
[![ML](https://img.shields.io/badge/TensorFlow-2.8.0-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A production-ready Deep Learning web application capable of classifying natural scenery images with high precision, featuring explainable AI via Grad-CAM visualizations.**

## 📋 Overview

The **Intel Image Classification Project** is a sophisticated computer vision application designed to automatically categorize images of natural environments. Built on top of a robust **Convolutional Neural Network (CNN)**, this project demonstrates end-to-end Machine Learning development—from model training to deployment via a **Flask** web interface.

One of the standout features of this application is its integration of **Explainable AI (XAI)**. Using **Grad-CAM (Gradient-weighted Class Activation Mapping)**, the system not only predicts what an image is but effectively "looks" at the image, generating heatmaps that highlight the specific regions contributing to the decision. This transparency is critical for building trust in AI systems.

## 🚀 Key Features

*   **🔍 High-Accuracy Classification**: Classifies images into 6 distinct categories:
    *   🏙️ Buildings
    *   🌲 Forest
    *   🧊 Glacier
    *   ⛰️ Mountain
    *   🌊 Sea
    *   🛣️ Street
*   **🧠 Explainable AI (Grad-CAM)**: Visualizes model attention with heatmap overlays, providing insights into *why* a specific prediction was made.
*   **🌐 Interactive Web Interface**: a responsive, user-friendly Flask-based frontend for easy image uploading and testing.
*   **⚡ Robust Backend**: Implements advanced error handling and flexible model loading strategies to ensure stability across different environments.
*   **📈 Performance**: Achieved **~81.5% validation accuracy** during training on the Intel Image Classification dataset.

## 🛠️ Tech Stack

*   **Language**: Python 3.x
*   **Deep Learning Framework**: TensorFlow / Keras
*   **Web Framework**: Flask
*   **Image Processing**: OpenCV, Pillow (PIL)
*   **Visualization**: Matplotlib, NumPy
*   **Containerization/Environment**: Conda

## 📂 Project Structure

```bash
📦 Intel_Image_Classification_project_flask
 ┣ 📂 models             # Serialized trained model files (.h5)
 ┣ 📂 static             # CSS, uploads, and generated heatmaps
 ┣ 📂 templates          # HTML templates for the web interface
 ┣ 📜 analyze_model.py     # Utility script to inspect model architecture
 ┣ 📜 app.py             # Main Flask application entry point
 ┣ 📜 gradcam.py         # Implementation of Grad-CAM algorithm
 ┣ 📜 requirements.txt   # Project dependencies
 ┗ 📜 README.md          # Project documentation
```

## 💻 Installation & Setup

Follow these steps to set up the project locally:

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/intel-image-classification.git
cd intel-image-classification
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Using Conda
conda create -n intel_env python=3.8
conda activate intel_env

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```
*The application will start on `http://localhost:5000`*

## 🎮 Usage Guide

1.  Open your browser and navigate to `http://localhost:5000`.
2.  Click the **"Choose File"** button to select an image from your local machine.
3.  Supported formats: `.jpg`, `.jpeg`, `.png`.
4.  Click **"Upload"** to process the image.
5.  View the **Prediction** (Class Name + Confidence Score) and the **Grad-CAM Visualization**.

## 📊 Model Performance

The Convolutional Neural Network was trained on the **Intel Image Classification** dataset.
*   **Training Accuracy**: ~88%
*   **Validation Accuracy**: ~81.5%

*Note metrics are approximate based on the latest training run logs.*

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or bug fixes, please open an issue or submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 👤 Author

**[Sachin Paunikar]**

*   💼 [LinkedIn](www.linkedin.com/in/sachin-paunikar-datascientists)
*   🐙 [GitHub](https://github.com/ImdataScientistSachin)
*   📧 [Email](mailto:ImdataScientistSachin@gmail.com)

---

*This project is designed for educational and portfolio purposes, showcasing skills in standard Computer Vision pipelines and Web Deployment.*
