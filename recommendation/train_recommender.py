
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
import os
from wide_and_deep_model import WideAndDeepModel

def prepare_data(data_dir='processed_data'):
    """
    Loads, preprocesses, and prepares data for the Wide & Deep model.
    """
    print("Preparing data for training...")
    reviews_df = pd.read_csv(os.path.join(data_dir, 'reviews.csv'))
    tours_df = pd.read_csv(os.path.join(data_dir, 'tours.csv'))
    
    # Merge to get features for each review
    df = pd.merge(reviews_df, tours_df, on='tour_id')
    
    # --- Define Feature Columns ---
    # For simplicity, we'll use a small subset of features.
    DEEP_COLS = ['user_id', 'tour_id']
    WIDE_COLS = ['city', 'state']
    
    # --- Target Variable ---
    # Create a binary target: 1 if the user liked it (e.g., > 3 stars), 0 otherwise.
    df['target'] = (df['stars_x'] > 3.5).astype(int)
    
    # --- Preprocessing ---
    # 1. Deep columns: Use LabelEncoder to create numerical IDs for embeddings
    for col in DEEP_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        joblib.dump(le, f'models/recommendation/{col}_encoder.pkl')

    # 2. Wide columns: Use OneHotEncoder
    preprocessor = ColumnTransformer(
        [('onehot', OneHotEncoder(handle_unknown='ignore'), WIDE_COLS)],
        remainder='passthrough'
    )
    X_wide = preprocessor.fit_transform(df[WIDE_COLS]).toarray()
    joblib.dump(preprocessor, 'models/recommendation/wide_preprocessor.pkl')
    
    X_deep = df[DEEP_COLS].values
    y = df['target'].values
    
    # --- Train/Test Split ---
    X_train_wide, X_test_wide, X_train_deep, X_test_deep, y_train, y_test = train_test_split(
        X_wide, X_deep, y, test_size=0.2, random_state=42
    )
    
    # --- Create PyTorch DataLoaders ---
    train_dataset = TensorDataset(torch.Tensor(X_train_wide), torch.LongTensor(X_train_deep), torch.Tensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test_wide), torch.LongTensor(X_test_deep), torch.Tensor(y_test))
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    
    # --- Model Dimensions ---
    num_users = df['user_id'].nunique()
    num_tours = df['tour_id'].nunique()
    wide_dim = X_train_wide.shape[1]
    
    return train_loader, test_loader, wide_dim, num_users, num_tours

def train_model(model, train_loader, test_loader, epochs=10, lr=0.001):
    """
    Main training loop for the model.
    """
    criterion = nn.BCELoss() # Binary Cross-Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        for x_wide, x_deep, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x_wide, x_deep)
            loss = criterion(outputs, y.unsqueeze(1))
            loss.backward()
            optimizer.step()
        
        # --- Validation ---
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x_wide, x_deep, y in test_loader:
                outputs = model(x_wide, x_deep)
                loss = criterion(outputs, y.unsqueeze(1))
                total_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).int()
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{epochs}, Val Loss: {avg_loss:.4f}, Val Accuracy: {accuracy:.2f}%')


if __name__ == '__main__':
    MODEL_DIR = 'models/recommendation'
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    train_loader, test_loader, wide_dim, num_users, num_tours = prepare_data(data_dir='data/processed_data')
    
    # --- Initialize and Train Model ---
    model = WideAndDeepModel(
        wide_dim=wide_dim,
        embedding_dims=[(num_users, 32), (num_tours, 16)],
        deep_layers=[128, 64]
    )
    
    print("\n--- Starting Model Training ---")
    train_model(model, train_loader, test_loader)
    
    # Save the final model
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'wide_and_deep_model.pth'))
    print(f"\nTraining complete. Model saved to '{MODEL_DIR}'.")
