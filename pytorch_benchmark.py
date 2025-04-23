import torch
import torch.nn.functional as F
import numpy as np
import time

# Hardcoded benchmark parameters
M = 10000
N = 9000
d = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def attention(Q, K, V):
    d = Q.size(-1)  # The dimensionality of the input vectors
    # Compute Q * K^T
    scores = torch.matmul(Q, K.T)  # Shape: (M, N)
    scaled_scores = scores / torch.sqrt(torch.tensor(d, dtype=torch.float32, device=Q.device))
    attention_weights = F.softmax(scaled_scores, dim=-1)  # Shape: (M, N)
    output = torch.matmul(attention_weights, V)  # Shape: (M, d)
    
    return output

def main():
    Q = np.zeros((M, d), dtype=np.float32)
    K = np.zeros((N, d), dtype=np.float32)
    V = np.zeros((N, d), dtype=np.float32)

    # Fill Q, K, V matrices with the same logic
    # used by the cuda main functions
    for i in range(M * d):
        Q.flat[i] = float(i) / (M * d)

    for i in range(N * d):
        K.flat[i] = float(i) * 2 / (N * d)
        V.flat[i] = float(i) * 3 / (N * d)

    # Convert to PyTorch tensors
    Q = torch.tensor(Q).to(device)
    K = torch.tensor(K).to(device)
    V = torch.tensor(V).to(device)

    start_time = time.time()
    O = attention(Q, K, V)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.6f} seconds")
    return end_time - start_time

if __name__ == "__main__":
    main()
