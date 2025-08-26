
import os
import torch
import joblib
from flask import Flask, request, jsonify
from PIL import Image

# --- Import Functions from Project Modules ---
# Note: Ensure these files and functions exist and are correctly defined.
from conversation.intent_recognizer import predict_intent
from vision.train_retrieval_encoder import load_encoder_model, extract_features
from vision.image_retrieval import find_similar_images
from recommendation.wide_and_deep_model import WideAndDeepModel

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load All Models on Startup ---
# This is a global dictionary to hold our models and data
models = {}

print("--- Loading all models into memory ---")
try:
    # 1. Conversational Models
    models['intent_encoder'] = joblib.load('models/conversation/intent_encoder.pkl')
    models['intent_classifier'] = joblib.load('models/conversation/intent_classifier.pkl')
    print("✅ Conversational models loaded.")

    # 2. Vision Models & Data
    models['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models['retrieval_encoder'] = load_encoder_model().to(models['device'])
    models['catalog_embeddings'] = joblib.load('models/vision/catalog_embeddings.pkl')
    models['catalog_paths'] = joblib.load('models/vision/catalog_paths.pkl')
    print("✅ Vision models loaded.")

    # 3. Recommendation Models (This is a simplified loading process)
    # For a real app, you'd also load encoders and preprocessors from train_recommender.py
    # models['wide_deep_recommender'] = WideAndDeepModel(...) # Initialize with correct dims
    # models['wide_deep_recommender'].load_state_dict(torch.load('models/recommendation/wide_and_deep_model.pth'))
    # models['wide_deep_recommender'].eval()
    print("☑️ Recommendation models would be loaded here (skipping for this example).")

except FileNotFoundError as e:
    print(f"❌ Error loading models: {e}. Please ensure all training scripts have been run.")
    models = None # Set to None to indicate failure

# --- API Endpoints ---

@app.route("/")
def index():
    return "<h1>AI-Powered E-commerce Assistant API</h1><p>Models loaded successfully!</p>" if models else "<h1>Error: Models not loaded.</h1>"

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint for the conversational agent."""
    if not models: return jsonify({"error": "Models are not loaded."}), 500
    
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({"error": "Query is a required field."}), 400
        
    # Predict the user's intent
    intent, confidence = predict_intent(
        query, 
        models['intent_encoder'], 
        models['intent_classifier']
    )
    
    # Simple logic based on intent
    response_text = f"Intent detected: {intent} (Confidence: {confidence:.2f})."
    if intent == 'search':
        response_text += " I can help you find products!"
    elif intent == 'greet':
        response_text += " Hello! How can I help you today?"
        
    return jsonify({
        "query": query,
        "intent": intent,
        "confidence": float(confidence),
        "response": response_text
    })

@app.route('/find_similar', methods=['POST'])
def find_similar():
    """Endpoint for finding visually similar items from an uploaded image."""
    if not models: return jsonify({"error": "Models are not loaded."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    
    image_file = request.files['image']
    
    # Extract features from the uploaded image
    query_features = extract_features(image_file, models['retrieval_encoder'], models['device'])
    
    if query_features is None:
        return jsonify({"error": "Could not process image."}), 500
        
    # Find similar items in our catalog
    similar_item_paths = find_similar_images(
        query_features,
        models['catalog_embeddings'],
        models['catalog_paths'],
        top_n=5
    )
    
    return jsonify({
        "status": "success",
        "similar_items": similar_item_paths
    })

# The /recommend endpoint for a Wide & Deep model is more complex
# as it requires generating features for many candidate items.
# This is a placeholder for how it might be structured.
@app.route('/recommend', methods=['GET'])
def recommend():
    """Endpoint for getting personalized recommendations."""
    if not models: return jsonify({"error": "Models are not loaded."}), 500
    
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "user_id parameter is required."}), 400
        
    # In a real system:
    # 1. Get all item IDs the user hasn't seen.
    # 2. For each (user, item) pair, generate wide & deep features using saved preprocessors.
    # 3. Run all pairs through the Wide & Deep model to get scores.
    # 4. Sort by score and return the top N items.
    
    return jsonify({
        "user_id": user_id,
        "recommendations": [
            "item_123", "item_456", "item_789"
        ],
        "note": "This is a placeholder for the Wide & Deep recommender."
    })


if __name__ == '__main__':
    app.run(debug=True, port=5001)
