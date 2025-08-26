
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import matplotlib.pyplot as plt
import os
from train_retrieval_encoder import load_encoder_model, extract_features
import random

def find_similar_images(query_features, catalog_features, catalog_paths, top_n=5):
    """
    Finds the top N most similar images to a query vector.
    
    NOTE: For very large catalogs, a library like FAISS would be much faster
    than cosine_similarity for this search.
    """
    # Calculate similarities
    similarities = cosine_similarity(query_features.reshape(1, -1), catalog_features)[0]
    
    # Get the indices of the top N most similar items
    top_indices = np.argsort(similarities)[::-1][1:top_n+1] # Exclude the query image itself
    
    # Return the paths of the top N images
    return [catalog_paths[i] for i in top_indices]

def display_results(query_path, result_paths, image_base_dir):
    """Displays the query image and the retrieval results."""
    plt.figure(figsize=(20, 5))
    
    # Display query image
    plt.subplot(1, 6, 1)
    plt.imshow(Image.open(os.path.join(image_base_dir, query_path)))
    plt.title("Query Image")
    plt.axis('off')
    
    # Display result images
    for i, res_path in enumerate(result_paths):
        plt.subplot(1, 6, i + 2)
        plt.imshow(Image.open(os.path.join(image_base_dir, res_path)))
        plt.title(f"Result {i+1}")
        plt.axis('off')
        
    plt.show()

if __name__ == '__main__':
    MODEL_DIR = 'models/vision'
    DATA_DIR = 'data/processed_data'
    IMAGE_DIR = os.path.join(DATA_DIR, 'images_resized')
    
    print("Loading retrieval models and data...")
    try:
        catalog_features = joblib.load(os.path.join(MODEL_DIR, 'catalog_embeddings.pkl'))
        catalog_paths = joblib.load(os.path.join(MODEL_DIR, 'catalog_paths.pkl'))
    except FileNotFoundError:
        print("Model files not found. Please run train_retrieval_encoder.py first.")
        exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_encoder_model().to(device)
    
    # --- Perform a sample retrieval ---
    # Pick a random image from our catalog to use as a query
    query_image_path_suffix = random.choice(catalog_paths)
    query_image_full_path = os.path.join(IMAGE_DIR, query_image_path_suffix)
    
    print(f"\nPerforming retrieval for query image: {query_image_path_suffix}")
    
    # Extract features from the query image
    query_features = extract_features(query_image_full_path, encoder, device)
    
    if query_features is not None:
        # Find similar images
        similar_images = find_similar_images(query_features, catalog_features, catalog_paths, top_n=5)
        
        # Display the results
        display_results(query_image_path_suffix, similar_images, IMAGE_DIR)
