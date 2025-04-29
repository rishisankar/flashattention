import torch
import torch.nn.functional as F
import numpy as np
import sys

# Hardcoded benchmark parameters
M = 8192
N = 8192
d = 32

error_threshold = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_expected_matrix():
    Q = torch.arange(M * d, dtype=torch.float32, device=device).reshape(M, d) / (M * d)
    K = torch.arange(N * d, dtype=torch.float32, device=device).reshape(N, d) * 2 / (N * d)
    V = torch.arange(N * d, dtype=torch.float32, device=device).reshape(N, d) * 3 / (N * d)
    Q = Q.unsqueeze(0).unsqueeze(0)  # (1, 1, M, d)
    K = K.unsqueeze(0).unsqueeze(0)  # (1, 1, N, d)
    V = V.unsqueeze(0).unsqueeze(0)  # (1, 1, N, d)

    O = F.scaled_dot_product_attention(Q, K, V)

    return O.reshape(M, d)

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
