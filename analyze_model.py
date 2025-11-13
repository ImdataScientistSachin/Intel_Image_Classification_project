import tensorflow as tf
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_model(model_path):
    """Analyze model architecture and configuration"""
    try:
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Print model summary
        print("\n=== Model Architecture ===")
        model.summary()
        
        # Analyze layers
        print("\n=== Layer Details ===")
        for idx, layer in enumerate(model.layers):
            print(f"\nLayer {idx}: {layer.name}")
            print(f"Type: {layer.__class__.__name__}")
            print(f"Input shape: {layer.input_shape}")
            print(f"Output shape: {layer.output_shape}")
            
            # Get layer config
            config = layer.get_config()
            print("Configuration:")
            for key, value in config.items():
                print(f"  {key}: {value}")
        
        # Test prediction
        print("\n=== Test Prediction ===")
        test_input = np.random.random((1, 150, 150, 3))
        predictions = model.predict(test_input)
        print(f"Test input shape: {test_input.shape}")
        print(f"Prediction output shape: {predictions.shape}")
        print(f"Prediction values range: [{np.min(predictions)}, {np.max(predictions)}]")
        
    except Exception as e:
        logger.error(f"Error analyzing model: {e}")
        raise

if __name__ == '__main__':
    MODEL_PATH = os.path.join('models', 'intel_image_Classifier_model.h5')
    analyze_model(MODEL_PATH)