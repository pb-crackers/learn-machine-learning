# Lab 4.3: Implementing Self-Attention from Scratch

## Objective

Build a simplified self-attention mechanism from scratch in NumPy, then implement multi-head attention, positional encoding, and visualize the attention weights to understand what the model "focuses on." By the end, you will have a working miniature transformer block.

## Learning Goals

- Implement scaled dot-product attention from first principles
- Understand the roles of Query, Key, and Value matrices
- Build multi-head attention by splitting inputs into parallel heads
- Implement sinusoidal positional encoding and understand why it is needed
- Visualize attention weight heatmaps to interpret model behavior
- Appreciate how these components combine into a full transformer block

## Prerequisites

- Python 3.8+
- NumPy
- Matplotlib

```bash
pip install numpy matplotlib seaborn
```

## Tasks

### Exercise 1: Scaled Dot-Product Attention

Implement the core attention formula:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

Steps:
1. Compute the dot product `QK^T` to get raw compatibility scores
2. Scale by `1/sqrt(d_k)` to prevent softmax saturation
3. Apply softmax row-wise to get attention weights (each row sums to 1)
4. Multiply weights by V to produce the output

Verify that:
- Output shape matches V's shape
- Each row of attention weights sums to 1.0
- Removing the scaling factor changes the weight distribution (try it!)

### Exercise 2: Multi-Head Attention

Split the input into multiple heads, compute attention for each independently, and concatenate the results:

1. Create separate W_Q, W_K, W_V projection matrices for each head
2. Project the input X into Q, K, V for each head
3. Apply scaled dot-product attention to each head
4. Concatenate all head outputs and project through W_O

Compare the attention patterns across different heads -- they should learn to focus on different relationships.

### Exercise 3: Visualize Attention Weights

Create heatmap visualizations showing which tokens attend to which other tokens:

1. Use a sample sentence like "The cat sat on the mat"
2. Compute attention weights using random (or learned) Q, K, V projections
3. Create a heatmap with tokens on both axes
4. Annotate cells with weight values
5. Visualize multiple heads side by side to see how they differ

### Exercise 4: Positional Encoding

Implement sinusoidal positional encoding:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

1. Generate the positional encoding matrix for positions 0 to 49
2. Visualize it as a heatmap -- notice the different frequency patterns
3. Show that nearby positions have similar encodings (high cosine similarity)
4. Show that the encoding for position `pos+k` can be linearly derived from position `pos`

## Expected Output

- A working `scaled_dot_product_attention()` function that processes a sequence of token embeddings
- A `multi_head_attention()` function that runs multiple attention heads in parallel
- Heatmap PNG files showing attention patterns for a sample sentence
- A positional encoding heatmap showing the sinusoidal patterns
- Understanding of how Q, K, V matrices transform the input to compute contextual representations

## Stretch Goals

- Add a causal mask to prevent positions from attending to future positions
- Implement a full transformer block (attention + FFN + residual + layer norm)
- Process a real sentence through your attention module using pre-trained word embeddings
- Compare the effect of different numbers of heads (1, 2, 4, 8) on attention patterns

## Starter Code

See `starter.py` for the skeleton implementation with TODO markers.
