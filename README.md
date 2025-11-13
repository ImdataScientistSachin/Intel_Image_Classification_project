# Intel Scene Classification with Explainability

**Project:** Intel Scene Classification with Explainability — a lightweight Flask web app that classifies scene images (buildings, forest, glacier, mountain, sea, street) using a pre-trained Keras/TensorFlow model and provides Grad-CAM visual explanations.

**Status:** Prototype / research demo

---

**What it is**
- **Purpose:** Classify images into six scene categories and produce interpretable Grad-CAM visualizations that highlight image regions influencing the model's prediction.
- **Stack:** Python, Flask, TensorFlow / Keras, Pillow, NumPy, Matplotlib, (optional) OpenCV for improved visuals.
- **Included assets:** Two model files in `models/` (`intel_image_Classifier_model.h5`, `intel_image_Classifier_V1_model.h5`), a simple web UI under `templates/` and `static/`, and utility scripts (`analyze_model.py`, `gradcam.py`).

**Quick highlights**
- Upload images through a web UI (`/`) — app returns predicted scene type and confidence.
- Generates Grad-CAM heatmaps saved to `static/uploads/` to explain the model's focus areas.
- Robust model-loading logic to mitigate common TF/HDF5 incompatibilities.

**Contents**
- `app.py`: Flask application (main entry point).
- `gradcam.py`: Grad-CAM implementation and visualization utilities.
- `analyze_model.py`: Script to inspect model architecture and run a sample prediction.
- `models/`: Saved model weights (`.h5`).
- `static/`: CSS and upload folder (`static/uploads/` where images and visualizations are stored).
- `templates/index.html`: Web UI template.
- `requirements.txt` / `environment.yml`: Python dependency manifests.

**Requirements**
- Python 3.8–3.10 recommended (project uses TensorFlow 2.8 in `requirements.txt`).
- GPU optional — CPU will work but inference is slower.

**Setup (recommended)**
1. Create and activate a virtual environment (pip):

```powershell
python -m venv venv
.\\venv\\Scripts\\Activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Or create the Conda environment (if you use Conda):

```powershell
conda env create -f environment.yml
conda activate Intel_Image_Classification
pip install -r requirements.txt  # optional: ensure pip extras are installed
```

**Run (development)**

```powershell
# From repository root
python app.py
# Open http://127.0.0.1:5000/ in your browser
```

**Usage**
- Open the web UI at `/`.
- Choose an image (JPG/PNG). The server will:
	- Save the uploaded file into `static/uploads/`.
	- Preprocess the image to `150x150` (see `preprocess_image` in `app.py`).
	- Run model prediction and return the predicted class and confidence.
	- Attempt to generate a Grad-CAM visualization saved to `static/uploads/gradcam_<image>.png` and display it in the UI.

**Programmatic / CLI tools**
- Analyze the model architecture and run a single synthetic prediction:

```powershell
python analyze_model.py
```

**Implementation notes & gotchas**
- Model input size: `150x150` RGB images. `app.py`'s `preprocess_image` resizes uploads accordingly.
- Model loading: `app.py` includes fallback loaders and `load_model_with_custom_handling()` to handle some HDF5 mismatches (batch_shape issues). If you encounter load errors, check the TensorFlow version used when the model was saved.
- Grad-CAM: `gradcam.py` will use OpenCV (`cv2`) when available for higher-quality overlays; otherwise it falls back to a Matplotlib-based simplified visualization. Install `opencv-python` for best results.
- File uploads: Uploaded files are saved with their original filename — consider sanitizing filenames in production to avoid conflicts and security issues.

**Security & production recommendations**
- Do not run the Flask dev server (used by `app.py`) behind public endpoints in production. Use a production-grade WSGI server (Gunicorn/Waitress) and a reverse proxy.
- Validate and sanitize uploaded filenames and restrict allowed content types.
- Enforce upload size limits and rate limits to avoid resource exhaustion.
- Consider storing model artifacts in a secure artifact store (S3, Azure Blob, etc.) rather than in the repo for larger deployments.

**Extending the project**
- Add unit tests and CI (GitHub Actions) to run linting and basic integration tests.
- Provide a `Dockerfile` to ease deployment and environment consistency.
- Add a small API endpoint that returns JSON predictions for programmatic access.
- Add model versioning and metadata (dataset reference, training config, license) to `models/`.

**Credits & Data**
- This project demonstrates model inference and basic explainability. If the models were trained on the Intel Image Classification dataset, cite the original dataset and any pre-processing steps in model metadata.

