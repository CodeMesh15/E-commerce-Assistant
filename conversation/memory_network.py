
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMemoryNetwork(nn.Module):
    """
    A simplified Memory Network for a multi-turn conversational agent.
    This model learns to pay attention to relevant past conversation steps (memory)
    to inform its next response.
    """
    def __init__(self, vocab_size, embedding_dim, response_size):
        super(SimpleMemoryNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Embedding layers for memory and queries
        self.embedding_A = nn.Embedding(vocab_size, embedding_dim) # For memory
        self.embedding_C = nn.Embedding(vocab_size, embedding_dim) # For response generation
        self.embedding_Q = nn.Embedding(vocab_size, embedding_dim) # For queries

        # Final linear layer to map output to a response
        self.output_layer = nn.Linear(embedding_dim, response_size)

    def forward(self, query, memory):
        """
        Forward pass of the Memory Network.
        
        Args:
            query (torch.Tensor): The current user query tensor. Shape: (batch_size, seq_len)
            memory (torch.Tensor): The conversation history. Shape: (batch_size, num_memories, seq_len)
        """
        # --- 1. Embed the query ---
        # Shape: (batch_size, seq_len, embedding_dim) -> (batch_size, embedding_dim)
        query_emb = self.embedding_Q(query).mean(dim=1)
        
        # --- 2. Embed the memory ---
        # Shape: (batch_size, num_memories, seq_len, embedding_dim) -> (batch_size, num_memories, embedding_dim)
        memory_emb_A = self.embedding_A(memory).mean(dim=2)
        
        # --- 3. Calculate Attention ---
        # This is the core of the model. We calculate how much the query matches each memory slot.
        # The dot product similarity is a common way to do this.
        # Shape: (batch_size, num_memories)
        attention_scores = torch.bmm(memory_emb_A, query_emb.unsqueeze(2)).squeeze(2)
        
        # Convert scores to a probability distribution (attention weights)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # --- 4. Generate Response Vector ---
        # First, get another embedding for the memory, used for response generation.
        memory_emb_C = self.embedding_C(memory).mean(dim=2)
        
        # Create the response vector by taking a weighted sum of memory embeddings
        # using the attention weights. This lets the model focus on important memories.
        # Shape: (batch_size, num_memories, embedding_dim) -> (batch_size, embedding_dim)
        response_vector = torch.bmm(attention_weights.unsqueeze(1), memory_emb_C).squeeze(1)
        
        # --- 5. Generate Final Output ---
        # Combine the response vector with the query embedding
        final_vector = response_vector + query_emb
        
        # Predict the final response
        output = self.output_layer(final_vector)
        
        return output, attention_weights


if __name__ == '__main__':
    # --- Example Demonstration ---
    
    # Define vocabulary and model parameters
    VOCAB_SIZE = 20
    EMBEDDING_DIM = 32
    NUM_RESPONSES = 5
    
    # Create dummy vocabulary mapping
    vocab = {"<pad>": 0, "i": 1, "need": 2, "a": 3, "blue": 4, "shirt": 5, "ok": 6, "how": 7, "about": 8, "in": 9, "small": 10, "size": 11}
    
    # Initialize the model
    model = SimpleMemoryNetwork(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, response_size=NUM_RESPONSES)
    
    # --- Simulate a conversation ---
    # Memory: ["i need a blue shirt", "ok"]
    memory_text = [[1, 2, 3, 4, 5], [6, 0, 0, 0, 0]] # Padded sentences
    
    # Current Query: "how about in small size"
    query_text = [7, 8, 9, 10, 11]
    
    # Convert to tensors (with a batch size of 1)
    memory_tensor = torch.LongTensor([memory_text])
    query_tensor = torch.LongTensor([query_text])
    
    # Perform a forward pass
    response_logits, attention = model(query_tensor, memory_tensor)
    
    print("--- Memory Network Demonstration ---")
    print(f"Memory: {memory_tensor.shape}")
    print(f"Query: {query_tensor.shape}")
    print("\nAttention Weights over Memories:")
    print(attention.detach().numpy())
    print("\nModel Output (Logits for each possible response):")
    print(response_logits.detach().numpy())
    print("\nNote: The high attention on the first memory slot shows the model is focusing on 'i need a blue shirt' to understand the context of the new query.")
