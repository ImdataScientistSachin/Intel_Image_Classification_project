import os
from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import logging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from gradcam import save_gradcam_visualization
import os.path as osp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'intel_image_Classifier_model.h5')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
# Load model
try:
    logger.info(f"Loading model from: {MODEL_PATH}")
    # First attempt: Try loading with custom_objects and skip_mismatch
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={'batch_shape': None},
            compile=False
        )
        logger.info("Model loaded successfully with custom_objects and compile=False")
    except ValueError as ve:
        logger.warning(f"First loading attempt failed: {ve}")
        # Second attempt: Try loading with tf.keras.experimental.load_from_saved_model if available
        try:
            logger.warning("Attempting experimental loading method...")
            # This is a more flexible loading method that might handle the batch_shape issue
            model = tf.saved_model.load(MODEL_PATH)
            logger.info("Model loaded successfully with experimental method")
        except Exception as e2:
            logger.error(f"Second loading attempt failed: {e2}")
            # Third attempt: Try the V1 model with the same approach
            alt_model_path = os.path.join(BASE_DIR, 'models', 'intel_image_Classifier_V1_model.h5')
            if os.path.exists(alt_model_path):
                logger.info(f"Attempting to load alternative model: {alt_model_path}")
                try:
                    # Try to load the model with a custom loader function
                    model = load_model_with_custom_handling(alt_model_path)
                    logger.info("Alternative model loaded successfully with custom handling")
                except Exception as e3:
                    logger.error(f"Alternative model loading failed: {e3}")
                    raise ValueError(f"All model loading attempts failed. Please recreate the model with compatible TensorFlow version.")
            else:
                raise ValueError(f"All model loading attempts failed and alternative model not found at {alt_model_path}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Add this custom model loading function at the top of your file
def load_model_with_custom_handling(model_path):
    """Load model with special handling for batch_shape parameter"""
    # Load the model file as an HDF5 file first
    import h5py
    import json
    import tensorflow as tf
    from tensorflow.python.keras.saving import hdf5_format
    
    # Open the HDF5 file
    with h5py.File(model_path, mode='r') as f:
        # Load the model config without the problematic batch_shape
        model_config = f.attrs.get('model_config')
        if model_config is None:
            raise ValueError("No model config found in the model file")
            
        if isinstance(model_config, bytes):
            model_config = model_config.decode('utf-8')
            
        model_config = json.loads(model_config)
        
        # Remove batch_shape from all input layers
        def remove_batch_shape(config):
            if 'config' in config and 'layers' in config['config']:
                for layer in config['config']['layers']:
                    if 'config' in layer and 'batch_shape' in layer['config']:
                        del layer['config']['batch_shape']
            return config
        
        model_config = remove_batch_shape(model_config)
        
        # Create a fresh model from the modified config
        from tensorflow.keras.models import model_from_json
        model = model_from_json(json.dumps(model_config))
        
        # Load weights
        hdf5_format.load_weights_from_hdf5_group(f['model_weights'], model.layers)
        
    return model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image to match model's requirements"""
    try:
        # Load image
        img = Image.open(image_path)
        logger.info(f"Original image size: {img.size}")

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            logger.info("Converted to RGB")

        # Resize to model's input size
        img = img.resize((150, 150))
        logger.info(f"Resized to: {img.size}")

        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        logger.info(f"Array shape: {img_array.shape}")

        # Model has built-in rescaling layer, so we don't normalize here
        img_array = np.expand_dims(img_array, axis=0)
        logger.info(f"Final shape: {img_array.shape}")

        return img_array

    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def make_prediction(img_array):
    """Make prediction with detailed logging"""
    try:
        # Verify input shape
        if img_array.shape != (1, 150, 150, 3):
            raise ValueError(f"Invalid input shape: {img_array.shape}")

        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        
        # Log probabilities for all classes
        logger.info("\nClass probabilities:")
        for class_name, prob in zip(CLASS_NAMES, predictions[0]):
            logger.info(f"{class_name}: {prob:.4f} ({prob*100:.2f}%)")

        # Get prediction
        predicted_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(predictions[0][predicted_index]) * 100

        logger.info(f"Final prediction: {predicted_class} ({confidence:.2f}%)")
        return predicted_class, confidence

    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise

# Add this after loading the model
# Find the last convolutional layer in the model
last_conv_layer_name = None
for layer in reversed(model.layers):
    if 'conv' in layer.name.lower():
        last_conv_layer_name = layer.name
        logger.info(f"Last convolutional layer identified: {last_conv_layer_name}")
        break
if not last_conv_layer_name:
    logger.warning("Could not identify last convolutional layer. Grad-CAM may not work.")
    last_conv_layer_name = model.layers[-3].name  # Fallback to third-to-last layer

# Modify the predict route to include Grad-CAM
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')

        if file and allowed_file(file.filename):
            try:
                # Save file
                filename = file.filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                logger.info(f"Saved file to: {filepath}")

                # Process and predict
                img_array = preprocess_image(filepath)
                predicted_class, confidence = make_prediction(img_array)
                
                # In the predict route, modify the Grad-CAM section
                
                # Generate Grad-CAM visualization
                gradcam_filename = f"gradcam_{osp.splitext(filename)[0]}.png"
                gradcam_filepath = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
                
                try:
                    save_gradcam_visualization(
                        filepath, 
                        gradcam_filepath, 
                        model, 
                        last_conv_layer_name, 
                        preprocess_image,
                        CLASS_NAMES
                    )
                    logger.info(f"Saved Grad-CAM visualization to: {gradcam_filepath}")
                    has_gradcam = True
                except Exception as e:
                    logger.error(f"Error generating Grad-CAM: {e}")
                    has_gradcam = False
                
                return render_template('index.html',
                                    prediction=predicted_class,
                                    confidence=f"{confidence:.2f}%",
                                    image_path=f"uploads/{filename}",
                                    gradcam_path=f"uploads/{gradcam_filename}" if has_gradcam else None)

            except Exception as e:
                logger.error(f"Error processing request: {e}")
                return render_template('index.html', error=str(e))
        else:
            return render_template('index.html', 
                                error='Please upload a JPG, JPEG or PNG image')

    return render_template('index.html')

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(debug=True, port=5000)