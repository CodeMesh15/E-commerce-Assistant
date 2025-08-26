
import torch
import torch.nn as nn

class WideAndDeepModel(nn.Module):
    """
    A Wide & Deep model for recommendation, combining a linear model and a DNN.

    The 'wide' component is a linear model that learns from sparse, cross-product features.
    It's good for memorizing simple, direct feature interactions.

    The 'deep' component is a feed-forward neural network that learns from dense embeddings.
    It's good for generalizing and discovering complex, unseen feature interactions.
    """
    def __init__(self, wide_dim, embedding_dims, deep_layers, deep_dropout=0.2):
        """
        Args:
            wide_dim (int): Number of features for the wide (linear) component.
            embedding_dims (list of tuples): List of (num_categories, embedding_dim)
                                             for each categorical feature in the deep path.
            deep_layers (list of ints): List of neuron counts for each hidden layer in the deep path.
            deep_dropout (float): Dropout probability for the deep path.
        """
        super(WideAndDeepModel, self).__init__()
        
        # --- Wide Component ---
        # A simple linear layer for the wide features
        self.wide_component = nn.Linear(wide_dim, 1)
        
        # --- Deep Component ---
        
        # 1. Embedding layers for categorical features
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_categories, emb_dim) for num_categories, emb_dim in embedding_dims
        ])
        
        total_embedding_dim = sum(emb_dim for _, emb_dim in embedding_dims)
        
        # 2. Deep neural network layers
        deep_network_layers = []
        input_dim = total_embedding_dim
        
        for layer_dim in deep_layers:
            deep_network_layers.append(nn.Linear(input_dim, layer_dim))
            deep_network_layers.append(nn.ReLU())
            deep_network_layers.append(nn.Dropout(deep_dropout))
            input_dim = layer_dim
            
        self.deep_component = nn.Sequential(*deep_network_layers)
        
        # --- Final Combination ---
        # The final output layer combines the outputs from both wide and deep components
        self.output_layer = nn.Linear(deep_layers[-1] + wide_dim, 1)

    def forward(self, x_wide, x_deep):
        """
        Forward pass of the model.
        
        Args:
            x_wide (torch.Tensor): Input for the wide component. Shape: (batch_size, wide_dim)
            x_deep (torch.Tensor): Input for the deep component. Shape: (batch_size, num_deep_features)
        """
        # --- Wide Path ---
        wide_output = self.wide_component(x_wide)
        
        # --- Deep Path ---
        # Get embeddings for each categorical feature
        deep_embeddings = [
            embedding_layer(x_deep[:, i]) for i, embedding_layer in enumerate(self.embedding_layers)
        ]
        
        # Concatenate embeddings
        concatenated_embeddings = torch.cat(deep_embeddings, dim=1)
        
        deep_output = self.deep_component(concatenated_embeddings)
        
        # --- Combine ---
        # Concatenate the outputs of both components and pass through the final layer
        combined = torch.cat([x_wide, deep_output], dim=1)
        
        final_output = self.output_layer(combined)
        
        # Use a sigmoid to get a probability-like score between 0 and 1
        return torch.sigmoid(final_output)

if __name__ == '__main__':
    # --- Example of how to initialize the model ---
    
    # Let's say we have:
    # - Wide features: 50 (from one-hot encoding categories etc.)
    # - Deep features: user_id (1000 users), item_id (500 items)
    
    WIDE_DIM = 50
    EMBEDDING_DIMS = [(1000, 32), (500, 16)] # (num_users, emb_size), (num_items, emb_size)
    DEEP_LAYERS = [128, 64, 32]
    
    model = WideAndDeepModel(
        wide_dim=WIDE_DIM,
        embedding_dims=EMBEDDING_DIMS,
        deep_layers=DEEP_LAYERS
    )
    
    print("--- Wide & Deep Model Architecture ---")
    print(model)
    
    # Create some dummy input to test the forward pass
    batch_size = 4
    dummy_wide = torch.randn(batch_size, WIDE_DIM)
    dummy_deep = torch.randint(0, 100, (batch_size, 2)) # 2 deep features: user_id, item_id
    
    output = model(dummy_wide, dummy_deep)
    print(f"\nDummy input shapes: wide={dummy_wide.shape}, deep={dummy_deep.shape}")
    print(f"Dummy output shape: {output.shape}")
    print(f"Dummy output values: {output.squeeze().tolist()}")
