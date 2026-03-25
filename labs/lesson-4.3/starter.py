"""
Lab 4.3: Implementing Self-Attention from Scratch
=================================================
Build the core attention mechanism that powers transformers.
"""

import numpy as np
import matplotlib.pyplot as plt


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ============================================================
# Exercise 1: Scaled Dot-Product Attention
# ============================================================

def scaled_dot_product_attention(Q, K, V):
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query matrix, shape (seq_len, d_k)
        K: Key matrix, shape (seq_len, d_k)
        V: Value matrix, shape (seq_len, d_v)

    Returns:
        output: Attention output, shape (seq_len, d_v)
        weights: Attention weights, shape (seq_len, seq_len)
    """
    d_k = K.shape[-1]

    # TODO: Step 1 - Compute attention scores: QK^T / sqrt(d_k)
    scores = None  # Your code here

    # TODO: Step 2 - Apply softmax to get attention weights
    weights = None  # Your code here

    # TODO: Step 3 - Multiply weights by V to get output
    output = None  # Your code here

    return output, weights


# Test Exercise 1
print("=" * 50)
print("Exercise 1: Scaled Dot-Product Attention")
print("=" * 50)

np.random.seed(42)
seq_len = 4
d_k = 8

# Create random Q, K, V matrices
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_k)

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Input shape: ({seq_len}, {d_k})")
print(f"Output shape: {output.shape}")
print(f"Weights shape: {weights.shape}")
print(f"Weights sum per row (should be ~1.0): {weights.sum(axis=-1)}")
print()


# ============================================================
# Exercise 2: Multi-Head Attention
# ============================================================

def multi_head_attention(X, n_heads, d_model):
    """
    Compute multi-head attention.

    Args:
        X: Input matrix, shape (seq_len, d_model)
        n_heads: Number of attention heads
        d_model: Model dimension

    Returns:
        output: Multi-head attention output, shape (seq_len, d_model)
        all_weights: List of attention weight matrices
    """
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    d_k = d_model // n_heads
    seq_len = X.shape[0]

    # Initialize weight matrices for Q, K, V projections
    # In practice these are learned; here we use random initialization
    W_Q = np.random.randn(n_heads, d_model, d_k) * 0.1
    W_K = np.random.randn(n_heads, d_model, d_k) * 0.1
    W_V = np.random.randn(n_heads, d_model, d_k) * 0.1
    W_O = np.random.randn(n_heads * d_k, d_model) * 0.1

    head_outputs = []
    all_weights = []

    for h in range(n_heads):
        # TODO: Project input to Q, K, V for this head
        Q_h = None  # Your code: X @ W_Q[h]
        K_h = None  # Your code: X @ W_K[h]
        V_h = None  # Your code: X @ W_V[h]

        # TODO: Apply scaled dot-product attention
        head_out, head_weights = scaled_dot_product_attention(Q_h, K_h, V_h)

        head_outputs.append(head_out)
        all_weights.append(head_weights)

    # TODO: Concatenate all heads and project
    concat = np.concatenate(head_outputs, axis=-1)  # (seq_len, n_heads * d_k)
    output = concat @ W_O  # (seq_len, d_model)

    return output, all_weights


# Test Exercise 2
print("=" * 50)
print("Exercise 2: Multi-Head Attention")
print("=" * 50)

d_model = 16
n_heads = 4
X = np.random.randn(seq_len, d_model)

output, all_weights = multi_head_attention(X, n_heads, d_model)
print(f"Input shape: {X.shape}")
print(f"Output shape: {output.shape}")
print(f"Number of heads: {len(all_weights)}")
print()


# ============================================================
# Exercise 3: Visualize Attention Weights
# ============================================================

def visualize_attention(weights, tokens, title="Attention Weights"):
    """
    Create a heatmap of attention weights.

    Args:
        weights: Attention weight matrix, shape (seq_len, seq_len)
        tokens: List of token strings
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(weights, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticklabels(tokens)
    ax.set_xlabel("Key (attending to)")
    ax.set_ylabel("Query (attending from)")
    ax.set_title(title)

    # Add text annotations
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            ax.text(j, i, f"{weights[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if weights[i, j] > 0.5 else "black")

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("attention_weights.png", dpi=150)
    print(f"Saved attention heatmap to attention_weights.png")
    plt.close()


print("=" * 50)
print("Exercise 3: Visualize Attention")
print("=" * 50)

tokens = ["The", "cat", "sat", "down"]

# Use the weights from Exercise 1 for visualization
visualize_attention(weights, tokens, "Self-Attention Weights")
print()


# ============================================================
# Exercise 4: Positional Encoding
# ============================================================

def positional_encoding(max_len, d_model):
    """
    Compute sinusoidal positional encoding.

    Args:
        max_len: Maximum sequence length
        d_model: Model dimension

    Returns:
        PE: Positional encoding matrix, shape (max_len, d_model)
    """
    PE = np.zeros((max_len, d_model))

    # TODO: Implement sinusoidal positional encoding
    # For each position pos and each dimension i:
    #   PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
    #   PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))

    position = np.arange(max_len)[:, np.newaxis]  # (max_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)

    return PE


print("=" * 50)
print("Exercise 4: Positional Encoding")
print("=" * 50)

PE = positional_encoding(50, 64)
print(f"Positional encoding shape: {PE.shape}")

# Visualize positional encoding
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(PE, cmap="RdBu", aspect="auto")
ax.set_xlabel("Dimension")
ax.set_ylabel("Position")
ax.set_title("Sinusoidal Positional Encoding")
plt.colorbar(im)
plt.tight_layout()
plt.savefig("positional_encoding.png", dpi=150)
print("Saved positional encoding visualization to positional_encoding.png")
plt.close()

print("\nDone! Check the generated PNG files for visualizations.")
