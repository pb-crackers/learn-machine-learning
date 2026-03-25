# Lab 4.2: Build a Text Generator with an LSTM in PyTorch

## Objective

Build a character-level LSTM language model in PyTorch that learns to generate text. Train it on a real text source and sample new text from the trained model.

## Learning Goals

- Understand how LSTMs process sequences through time steps
- Prepare text data for character-level language modeling (encoding, batching, sequencing)
- Build an LSTM model in PyTorch using `nn.Module`
- Implement a training loop with sequence data and cross-entropy loss
- Generate text by sampling from the model's output distribution
- Experiment with temperature to control output randomness

## Prerequisites

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install torch numpy matplotlib
```

## Dataset

This lab uses a built-in sample text (a passage from public-domain literature). You can substitute any plain text file of your choice -- the longer the better for quality results.

## Tasks

### Task 1: Prepare the Text Data
- Load the text and build character-to-index and index-to-character mappings
- Create input/target sequence pairs for training (each target is the input shifted by one character)
- Implement a PyTorch `Dataset` that returns fixed-length chunks of text as tensors

### Task 2: Build the LSTM Model
- Create an `nn.Module` subclass with:
  - An embedding layer to convert character indices to dense vectors
  - One or more LSTM layers with configurable hidden size
  - A fully connected output layer mapping hidden states to character logits
- Initialize the hidden state properly

### Task 3: Train the Model
- Implement the training loop with:
  - Cross-entropy loss on character predictions
  - Adam optimizer
  - Gradient clipping to prevent exploding gradients
  - Periodic logging of loss and a generated text sample
- Train for enough epochs to see coherent text emerge

### Task 4: Generate Text
- Implement a generation function that:
  - Takes a seed string as input
  - Feeds it through the model character by character to build hidden state
  - Samples the next character from the output distribution
  - Repeats for a desired number of characters
- Experiment with temperature scaling (low = conservative, high = creative)

### Task 5: Experiment and Analyze
- Generate text at temperatures 0.2, 0.5, 0.8, and 1.2
- Plot the training loss curve
- Try increasing/decreasing the hidden size or number of LSTM layers
- Discuss how sequence length affects what the model can learn

## Expected Output

```
Vocabulary: 65 unique characters

Epoch 1/50, Loss: 3.1284
  Sample: "tHe qR&xzL mP!k..."

Epoch 25/50, Loss: 1.4521
  Sample: "the king of the land was..."

Epoch 50/50, Loss: 1.1037
  Sample: "the prince rode forth into the great forest..."

--- Generation (temperature=0.5) ---
"In the beginning there was a great kingdom that stretched
 across the mountains and the sea. The people of the land..."

--- Generation (temperature=1.0) ---
"In the beginning dede was a grest kinding thot stwetceld
 acress nhe mount and sev. Tre paople on tge lant..."
```

(Actual output depends on the training corpus and random seed.)

## Stretch Goals

- Switch to word-level tokenization and compare with character-level
- Add dropout between LSTM layers and measure the effect on overfitting
- Implement beam search decoding instead of random sampling
- Train on a larger corpus (e.g., download a full book from Project Gutenberg)
- Add a classification head and fine-tune the LSTM for sentiment analysis
