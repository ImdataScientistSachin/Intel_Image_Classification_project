import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging

# Configure logging if not already configured
logger = logging.getLogger(__name__)

# Try to import OpenCV, but provide fallback
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    logger.warning("OpenCV (cv2) could not be imported. Grad-CAM visualization will be limited.")
    OPENCV_AVAILABLE = False

"""
- It demonstrates your understanding of model interpretability, a critical aspect of responsible AI
- It provides visual evidence of how your model works, making your project more engaging
- It shows technical depth beyond basic model implementation
"""

# Add this near the top of the file, after the imports
import matplotlib.font_manager as fm

# Then before creating any plots
plt.rcParams['font.family'] = 'Segoe UI Emoji'  # Or another emoji-compatible font

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap for the specified image and model.
    
    Args:
        img_array: Input image as a preprocessed array
        model: The trained model
        last_conv_layer_name: Name of the last convolutional layer
        pred_index: Index of the predicted class (None for highest scoring class)
        
    Returns:
        Superimposed visualization and raw heatmap
    """
    # Create a model that maps the input image to the activations of the last conv layer
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute gradient of top predicted class with respect to last conv layer output
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # Gradient of the predicted class with respect to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Vector of mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight output feature map with gradient importance
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    return heatmap

def create_gradcam_visualization(image_path, model, last_conv_layer_name, preprocess_func, class_names):
    """Create a Grad-CAM visualization for an image.
    
    Args:
        image_path: Path to the input image
        model: The trained model
        last_conv_layer_name: Name of the last convolutional layer
        preprocess_func: Function to preprocess the image
        class_names: List of class names
        
    Returns:
        Original image, heatmap overlay, and prediction information
    """
    # Load and preprocess image
    img_array = preprocess_func(image_path)
    
    # Make prediction
    preds = model.predict(img_array)
    pred_index = np.argmax(preds[0])
    pred_class = class_names[pred_index]
    confidence = float(preds[0][pred_index]) * 100
    
    # Generate heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    
    if not OPENCV_AVAILABLE:
        # Create a simple placeholder image if OpenCV is not available
        logger.warning("OpenCV not available. Using simplified visualization.")
        # Return a simple representation using matplotlib
        img = plt.imread(image_path)
        return img, img, pred_class, confidence
    
    # Load original image for display
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cm.jet(heatmap)[..., :3] * 255
    heatmap = heatmap.astype(np.uint8)
    
    # Superimpose heatmap on original image
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    return img, superimposed_img, pred_class, confidence

def save_gradcam_visualization(image_path, output_path, model, last_conv_layer_name, preprocess_func, class_names):
    """Save Grad-CAM visualization to a file.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the visualization
        model: The trained model
        last_conv_layer_name: Name of the last convolutional layer
        preprocess_func: Function to preprocess the image
        class_names: List of class names
        
    Returns:
        Path to the saved visualization
    """
    # Generate visualization
    img, superimposed_img, pred_class, confidence = create_gradcam_visualization(
        image_path, model, last_conv_layer_name, preprocess_func, class_names
    )
    
    # Create figure with subplots - using a more modern style
    plt.style.use('seaborn-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor('#f0f8ff')  # Light blue background
    
    # Display original image with a slight border
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_color('#3498db')  # Blue border
        spine.set_linewidth(2)
    
    # Display heatmap overlay with a creative title
    ax2.imshow(superimposed_img)
    ax2.set_title(f'AI Vision Heatmap', fontsize=14, fontweight='bold')
    ax2.axis('off')
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_color('#3498db')  # Blue border
        spine.set_linewidth(2)
    
    # Add a fun, creative title
    # Option A: Using stars (most compatible)
    plt.suptitle(f'** How AI Sees Your {pred_class} Scene **', fontsize=18, fontweight='bold', color='#2c3e50')
    
    # Option B: Using standard symbols
    plt.suptitle(f'★ How AI Sees Your {pred_class} Scene ★', fontsize=18, fontweight='bold', color='#2c3e50')
    
    # Option C: No symbols
    plt.suptitle(f'How AI Sees Your {pred_class} Scene', fontsize=18, fontweight='bold', color='#2c3e50')
    
    # Adjust layout to make more room for text at bottom and move second image down
    plt.subplots_adjust(bottom=0.35, hspace=0.3)
    

    
    # Create a more visually appealing explanation box
    explanation_text = (
        f"AI VISION DECODED!"
        f"\n\nThe AI is {confidence:.1f}% sure this is a {pred_class} scene!"
        f"\n\n[Red areas]: \"These parts scream '{pred_class}' to me!\""
        f"\n[Blue areas]: \"These didn't help me much in my decision.\""
        f"\n\nThis heat vision reveals how the AI's 'brain' focuses on specific features"
        f"\nto recognize scenes - just like how you might notice trees in a forest or"
        f"\nbuildings in a cityscape! Cool, right?"
    )
    
    # Create a more attractive box with gradient-like effect for the main explanation
    props = dict(
        boxstyle='round,pad=1.2', 
        facecolor='#e8f4f8',  # Light blue background
        edgecolor='#3498db',   # Blue border
        linewidth=3,
        alpha=0.9
    )
    
    # Place text below both images with improved styling
    plt.figtext(0.5, 0.02, explanation_text, ha='center', fontsize=12, va='bottom', 
                multialignment='center', bbox=props, fontweight='medium', color='#2c3e50')
    
    # Create a more creative and compact style for the classification display
    pred_props = dict(
        boxstyle='round4,pad=0.6',  # Rounded corners with smaller padding
        facecolor='#2c3e50',  # Dark blue background for contrast
        edgecolor='#3498db',  # Bright blue border
        linewidth=2,
        alpha=0.85  # Slightly transparent
    )

    # Add a professional attribution line with improved styling
    plt.figtext(0.5, -0.02, f"CLASSIFICATION: {pred_class.upper()} | CONFIDENCE: {confidence:.1f}%", 
                ha='center', fontsize=11, color='#ffffff', fontweight='bold',
                bbox=pred_props)  # Using the pred_props dictionary for styling
    
    # Add a small attribution/info line with better styling to match the image you shared
    pred_props = dict(
        boxstyle='round,pad=0.8',
        facecolor='#f8f9fa',  # Light gray background
        edgecolor='#95a5a6',  # Subtle border
        linewidth=1,
        alpha=0.9
    )
    

    # Add a professional attribution line with improved styling
    plt.figtext(0.5, -0.02, f"CLASSIFICATION: {pred_class.upper()} | CONFIDENCE: {confidence:.1f}%", 
                ha='center', fontsize=11, color='#2c3e50', fontweight='bold',
                bbox=dict(facecolor='#f5f5f5', alpha=0.95, pad=6, 
                          edgecolor='#34495e', linewidth=1.5, boxstyle='round,pad=0.6'))
    
    # Save figure with tight layout to ensure all elements are visible
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    return output_path