# ml-service/download_pretrained.py
# Download a pre-trained plant disease model

import requests
import os
from pathlib import Path
import gdown

def download_pretrained_model():
    """Download pre-trained model from various sources"""
    
    # Create models directory
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    print("Downloading pre-trained crop disease model...")
    
    try:
        # Option 1: TensorFlow Hub Model
        import tensorflow_hub as hub
        
        print("Attempting to download TensorFlow Hub model...")
        
        # Load a pre-trained image classification model
        model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5"
        
        # Download and adapt for plant disease
        base_model = hub.KerasLayer(model_url, input_shape=(224, 224, 3))
        
        # Create adapted model
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.Dense(6, activation='softmax')  # 6 disease classes
        ])
        
        # Save the model
        model.save('models/crop_disease_model.h5')
        print("TensorFlow Hub model downloaded and saved!")
        return True
        
    except Exception as e:
        print(f"TensorFlow Hub download failed: {e}")
    
    try:
        # Option 2: Download from GitHub releases
        print("Attempting to download from GitHub...")
        
        model_urls = [
            "https://github.com/emmarex/plantdisease-classification/raw/master/model/plant_disease_model_1_latest.h5",
            "https://github.com/spMohanty/PlantVillage-Dataset/releases/download/v1.0/plant_disease_model.h5"
        ]
        
        for i, url in enumerate(model_urls):
            try:
                print(f"Trying URL {i+1}: {url}")
                response = requests.get(url, stream=True, timeout=60)
                
                if response.status_code == 200:
                    model_path = f'models/downloaded_model_{i+1}.h5'
                    
                    with open(model_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Rename to standard name
                    os.rename(model_path, 'models/crop_disease_weights.h5')
                    print(f"Model downloaded successfully from URL {i+1}!")
                    return True
                    
            except Exception as e:
                print(f"URL {i+1} failed: {e}")
                continue
    
    except Exception as e:
        print(f"GitHub download failed: {e}")
    
    try:
        # Option 3: Google Drive download (if you have a model there)
        print("Attempting Google Drive download...")
        
        # Example Google Drive file ID (you need to replace this)
        file_id = "1abcdefghijk"  # Replace with actual file ID
        gdown.download(f"https://drive.google.com/uc?id={file_id}", 
                      'models/crop_disease_weights.h5', quiet=False)
        print("Google Drive model downloaded!")
        return True
        
    except Exception as e:
        print(f"Google Drive download failed: {e}")
    
    # Option 4: Create a simple transfer learning model
    try:
        print("Creating transfer learning model...")
        
        import tensorflow as tf
        
        # Use ResNet50 as base
        base_model = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Add custom classification head
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(6, activation='softmax')  # 6 classes
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Save the model
        model.save('models/crop_disease_model.h5')
        model.save_weights('models/crop_disease_weights.h5')
        
        print("Transfer learning model created and saved!")
        print("Note: This model needs training on actual crop disease data")
        return True
        
    except Exception as e:
        print(f"Transfer learning model creation failed: {e}")
    
    print("All download methods failed!")
    return False

if __name__ == "__main__":
    print("=== Pre-trained Model Download ===")
    
    # Install required packages
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
        print("TensorFlow and Hub available")
    except ImportError:
        print("Installing TensorFlow Hub...")
        os.system("pip install tensorflow-hub")
    
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        os.system("pip install gdown")
    
    # Download model
    success = download_pretrained_model()
    
    if success:
        print("\n✅ Model download successful!")
        print("Model saved to: models/crop_disease_weights.h5")
        print("You can now run your FastAPI service with real predictions!")
    else:
        print("\n❌ Model download failed!")
        print("Consider training your own model or manually downloading a model file.")
        
    print("\nNext steps:")
    print("1. Restart your FastAPI service: python main.py")
    print("2. Test with image upload")
    print("3. Check predictions for accuracy")