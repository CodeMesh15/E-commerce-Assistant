
import torch
import torchvision.transforms as T
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import os

def load_encoder_model():
    """
    Loads a pre-trained ResNet-50 model and modifies it to be a feature extractor.
    """
    model = resnet50(pretrained=True)
    # Remove the final classification layer
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    return model

def extract_features(image_path, model, device):
    """
    Extracts a feature vector from a single image.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        
        # Preprocess the image
        preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = model(input_batch)
        
        # Flatten the features to a 1D vector
        return features.squeeze().cpu().numpy()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def create_embeddings_for_catalog(processed_data_dir, model_dir='models/vision'):
    """
    Iterates through the image catalog, extracts features for each, and saves them.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    encoder = load_encoder_model().to(device)
    
    # Load the master annotation file to get image paths
    annotations_path = os.path.join(processed_data_dir, 'master_annotations.csv')
    if not os.path.exists(annotations_path):
        print(f"'{annotations_path}' not found. Please run data/preprocess_images.py first.")
        return

    df = pd.read_csv(annotations_path)
    image_paths = df['image_name'].tolist()
    
    all_features = []
    valid_paths = []
    
    image_base_dir = os.path.join(processed_data_dir, 'images_resized')

    for img_path_suffix in tqdm(image_paths, desc="Creating Embeddings"):
        full_path = os.path.join(image_base_dir, img_path_suffix)
        if os.path.exists(full_path):
            features = extract_features(full_path, encoder, device)
            if features is not None:
                all_features.append(features)
                valid_paths.append(img_path_suffix)

    # Save the embeddings and corresponding paths
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(np.array(all_features), os.path.join(model_dir, 'catalog_embeddings.pkl'))
    joblib.dump(valid_paths, os.path.join(model_dir, 'catalog_paths.pkl'))
    
    print(f"\nEmbeddings for {len(valid_paths)} images created and saved.")

if __name__ == '__main__':
    create_embeddings_for_catalog(processed_data_dir='data/processed_data')
