import torch
import torch.nn.functional as F
import numpy as np
import sys

# Hardcoded benchmark parameters
M = 10000
N = 9000
d = 32

error_threshold = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def attention(Q, K, V):
    d = Q.size(-1)  # The dimensionality of the input vectors
    # Compute Q * K^T
    scores = torch.matmul(Q, K.T)  # Shape: (M, N)
    scaled_scores = scores / torch.sqrt(torch.tensor(d, dtype=torch.float32, device=Q.device))
    attention_weights = F.softmax(scaled_scores, dim=-1)  # Shape: (M, N)
    output = torch.matmul(attention_weights, V)  # Shape: (M, d)
    
    return output

def compute_expected_matrix():
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

    O = attention(Q, K, V)

    return O

def main(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        actual_O = torch.tensor([[float(x) for x in line.split()] for line in lines], dtype=torch.float32).to(device)

    # Compute the actual O matrix
    expected_O = compute_expected_matrix()

    # Ensure the shapes match
    if expected_O.shape != actual_O.shape:
        print("Shape mismatch between expected and actual matrices.")
        return

    # Compute differences
    abs_error = torch.abs(actual_O - expected_O)
    rel_error = abs_error / (torch.abs(expected_O) + 1e-8)  # Add epsilon to avoid division by zero

    max_abs_error = torch.max(abs_error).item()
    max_rel_error = torch.max(rel_error).item()

    # Print results
    print(f"Max absolute error: {max_abs_error}")
    print(f"Max relative error: {max_rel_error}")

    
    if max_abs_error <= error_threshold and max_rel_error <= error_threshold:
        print("Verification passed: Output is within acceptable error thresholds.")
    else:
        print("Verification failed: Output exceeds acceptable error thresholds.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_output.py <filepath>")
    else:
        main(sys.argv[1])
