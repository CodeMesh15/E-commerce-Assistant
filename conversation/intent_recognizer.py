
import joblib
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_intent_model(model_dir='models/conversation'):
    """
    Trains and saves a sentence transformer and an intent classification model.
    """
    print("Training intent recognition model...")
    
    # --- 1. Create a synthetic dataset ---
    # In a real project, this would come from a CSV file.
    data = [
        ("hello", "greet"),
        ("hi there", "greet"),
        ("good morning", "greet"),
        ("I'm looking for a blue t-shirt", "search"),
        ("do you have any summer dresses?", "search"),
        ("show me some black pants", "search"),
        ("can you recommend something for me?", "ask_recommendation"),
        ("what's popular right now?", "ask_recommendation"),
        ("suggest an outfit", "ask_recommendation"),
        ("thanks, that's all", "goodbye"),
        ("bye", "goodbye"),
        ("thank you, goodbye", "goodbye")
    ]
    df = pd.DataFrame(data, columns=['text', 'intent'])
    
    # --- 2. Create Text Embeddings ---
    print("Loading sentence transformer and creating embeddings...")
    # Using a lightweight, high-performance model
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = encoder.encode(df['text'].tolist(), show_progress_bar=True)
    
    # --- 3. Train the Classifier ---
    print("Training the SVM classifier...")
    X_train, X_test, y_train, y_test = train_test_split(embeddings, df['intent'], test_size=0.2, random_state=42)
    
    # An SVM is a good choice for this type of high-dimensional data
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(X_train, y_train)
    
    # --- 4. Evaluate and Save ---
    print("\nEvaluating model performance:")
    predictions = classifier.predict(X_test)
    print(classification_report(y_test, predictions))
    
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(encoder, os.path.join(model_dir, 'intent_encoder.pkl'))
    joblib.dump(classifier, os.path.join(model_dir, 'intent_classifier.pkl'))
    
    print(f"Intent recognition models saved to '{model_dir}'")
    return encoder, classifier

def predict_intent(text, encoder, classifier):
    """
    Predicts the intent of a given text query.
    """
    # Create embedding for the input text
    embedding = encoder.encode([text])
    
    # Predict the intent
    intent = classifier.predict(embedding)[0]
    confidence = classifier.predict_proba(embedding).max()
    
    return intent, confidence

if __name__ == '__main__':
    # Train the model (this only needs to be done once)
    encoder_model, classifier_model = train_intent_model()
    
    # Example of loading and using the models
    print("\n--- Testing the trained models ---")
    loaded_encoder = joblib.load('models/conversation/intent_encoder.pkl')
    loaded_classifier = joblib.load('models/conversation/intent_classifier.pkl')
    
    test_queries = [
        "hi",
        "I need a new pair of jeans",
        "what should I wear?",
        "thanks"
    ]
    
    for query in test_queries:
        pred_intent, pred_confidence = predict_intent(query, loaded_encoder, loaded_classifier)
        print(f"Query: '{query}' -> Intent: {pred_intent} (Confidence: {pred_confidence:.2f})")
