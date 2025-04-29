import torch
import torch.nn.functional as F
import time

# Hardcoded benchmark parameters
M = 8192
N = 8192
d = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    torch.backends.cuda.enable_flash_sdp(True)

    # Initialize Q, K, V directly as PyTorch tensors on the GPU
    Q = torch.arange(M * d, dtype=torch.float32, device=device).reshape(M, d) / (M * d)
    K = torch.arange(N * d, dtype=torch.float32, device=device).reshape(N, d) * 2 / (N * d)
    V = torch.arange(N * d, dtype=torch.float32, device=device).reshape(N, d) * 3 / (N * d)

    # Reshape to match the expected input shape for scaled_dot_product_attention
    Q = Q.unsqueeze(0).unsqueeze(0)  # (1, 1, M, d)
    K = K.unsqueeze(0).unsqueeze(0)  # (1, 1, N, d)
    V = V.unsqueeze(0).unsqueeze(0)  # (1, 1, N, d)

    with torch.no_grad():
        start_time = time.time()
        O = F.scaled_dot_product_attention(Q, K, V)
        end_time = time.time()

    print(f"Time taken: {end_time - start_time:.6f} seconds")
    return end_time - start_time

if __name__ == "__main__":
    main()
