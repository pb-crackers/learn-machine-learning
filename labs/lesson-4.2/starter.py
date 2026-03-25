"""
Lab 4.2: Build a Text Generator with an LSTM in PyTorch
========================================================

Build a character-level LSTM language model that learns to
generate text. Train it on a sample corpus and experiment
with temperature-based sampling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Sample Corpus
# ============================================================

SAMPLE_TEXT = """\
Once upon a time in a land far away, there lived a wise old king who ruled over a \
peaceful kingdom. The kingdom was nestled between tall mountains and a sparkling sea. \
The people of the land were happy and prosperous, for the king was just and fair in \
all his dealings. Every morning the king would walk through the village and greet his \
subjects, asking about their lives and listening to their concerns. The children would \
run alongside him, laughing and playing, while the elders sat in the shade and told \
stories of days gone by. In the great castle at the heart of the kingdom, the king \
held court each afternoon, settling disputes and making laws that served the common \
good. His advisors were learned men and women who had traveled far and wide, bringing \
knowledge from distant lands. The royal library contained thousands of books on every \
subject imaginable, from the movements of the stars to the secrets of the deep ocean. \
The king himself was a scholar who spent his evenings reading by candlelight, always \
seeking to understand the world around him. One day a stranger arrived at the gates \
of the kingdom, carrying a mysterious book that was said to contain the wisdom of the \
ancients. The stranger spoke of great discoveries and inventions that could transform \
the kingdom forever. The king welcomed the stranger into the castle and together they \
studied the ancient text, unlocking secrets that had been lost for centuries. With this \
new knowledge the kingdom flourished even more, building great bridges and towers, \
creating beautiful works of art, and discovering new ways to heal the sick. The people \
celebrated their good fortune with festivals and feasts that lasted for days. Musicians \
played in the streets, artists painted murals on the walls, and poets composed verses \
that would be remembered for generations. The king knew that knowledge was the greatest \
treasure of all, and he decreed that every child in the kingdom should learn to read \
and write. Schools were built in every village, and teachers came from far and wide to \
share their wisdom. The kingdom became known throughout the world as a beacon of \
learning and enlightenment, and travelers came from distant shores to study in its \
great libraries and universities. And so the kingdom prospered for many years, guided \
by the wisdom of its beloved king and the curiosity of its people.\
"""


# ============================================================
# Task 1: Data Preparation
# ============================================================

class CharDataset(Dataset):
    """A character-level text dataset for language modeling.

    Each sample is a pair (input_seq, target_seq) where target_seq
    is input_seq shifted by one character.

    Parameters
    ----------
    text : str
        The raw text corpus.
    seq_length : int
        Length of each training sequence.
    """

    def __init__(self, text, seq_length=64):
        self.text = text
        self.seq_length = seq_length

        # Build character vocabulary
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        # Encode the entire text as a list of integers
        self.encoded = [self.char_to_idx[ch] for ch in text]

        print(f"  Text length: {len(text):,} characters")
        print(f"  Vocabulary: {self.vocab_size} unique characters")
        print(f"  Characters: {''.join(self.chars)}")

    def __len__(self):
        # Number of sequences we can extract
        return len(self.encoded) - self.seq_length

    def __getitem__(self, idx):
        """Return (input_sequence, target_sequence) as LongTensors.

        The target is the input shifted by one position:
          input:  "Hello Worl"
          target: "ello World"
        """
        # TODO: Extract a chunk of seq_length characters starting at idx
        # TODO: The target is the same chunk shifted by one position
        # input_seq = torch.tensor(self.encoded[idx : idx + self.seq_length],
        #                          dtype=torch.long)
        # target_seq = torch.tensor(self.encoded[idx + 1 : idx + 1 + self.seq_length],
        #                           dtype=torch.long)
        # return input_seq, target_seq

        # Placeholder -- replace with the code above
        input_seq = torch.zeros(self.seq_length, dtype=torch.long)
        target_seq = torch.zeros(self.seq_length, dtype=torch.long)
        return input_seq, target_seq


# ============================================================
# Task 2: Build the LSTM Model
# ============================================================

class CharLSTM(nn.Module):
    """Character-level LSTM language model.

    Parameters
    ----------
    vocab_size : int
        Number of unique characters.
    embed_size : int
        Dimensionality of character embeddings.
    hidden_size : int
        Number of LSTM hidden units.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability between LSTM layers.
    """

    def __init__(self, vocab_size, embed_size=64, hidden_size=256,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # TODO: Define the layers
        # 1. Embedding layer: maps character indices to dense vectors
        # self.embedding = nn.Embedding(vocab_size, embed_size)

        # 2. LSTM: processes sequences of embeddings
        # self.lstm = nn.LSTM(
        #     input_size=embed_size,
        #     hidden_size=hidden_size,
        #     num_layers=num_layers,
        #     dropout=dropout if num_layers > 1 else 0,
        #     batch_first=True,
        # )

        # 3. Fully connected output layer: maps hidden state to logits
        # self.fc = nn.Linear(hidden_size, vocab_size)

        pass  # Remove this once you add the layers above

    def forward(self, x, hidden=None):
        """Forward pass through the model.

        Parameters
        ----------
        x : Tensor, shape (batch_size, seq_length)
            Input character indices.
        hidden : tuple of Tensors, optional
            Previous (h_0, c_0) hidden state. If None, initialized to zeros.

        Returns
        -------
        logits : Tensor, shape (batch_size, seq_length, vocab_size)
            Raw predictions for each character position.
        hidden : tuple of Tensors
            Updated hidden state (h_n, c_n).
        """
        # TODO: Implement the forward pass
        # 1. Embed the input characters
        # embedded = self.embedding(x)         # (batch, seq, embed_size)

        # 2. Pass through the LSTM
        # output, hidden = self.lstm(embedded, hidden)  # (batch, seq, hidden)

        # 3. Map to vocabulary logits
        # logits = self.fc(output)             # (batch, seq, vocab_size)

        # return logits, hidden

        # Placeholder -- returns random logits
        batch_size, seq_length = x.shape
        logits = torch.zeros(batch_size, seq_length, 1)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        """Create initial hidden state filled with zeros.

        Parameters
        ----------
        batch_size : int
        device : torch.device

        Returns
        -------
        (h_0, c_0) : tuple of Tensors
            Each has shape (num_layers, batch_size, hidden_size).
        """
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                           device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                           device=device)
        return (h_0, c_0)


# ============================================================
# Task 3: Training Loop
# ============================================================

def train_model(model, dataloader, device, epochs=50, lr=0.002,
                clip=1.0, print_every=5, dataset=None):
    """Train the character-level LSTM.

    Parameters
    ----------
    model : CharLSTM
    dataloader : DataLoader
    device : torch.device
    epochs : int
    lr : float
        Learning rate.
    clip : float
        Max gradient norm for clipping.
    print_every : int
        Print loss and sample every N epochs.
    dataset : CharDataset, optional
        Used for generating sample text during training.

    Returns
    -------
    losses : list of float
        Average loss per epoch.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = inputs.size(0)

            # Initialize hidden state for each batch
            hidden = model.init_hidden(batch_size, device)

            # TODO: Forward pass
            # Detach hidden state from previous batch's computation graph
            # hidden = (hidden[0].detach(), hidden[1].detach())
            # logits, hidden = model(inputs, hidden)

            # TODO: Compute loss
            # Reshape logits and targets for cross-entropy:
            #   logits: (batch * seq_length, vocab_size)
            #   targets: (batch * seq_length,)
            # loss = criterion(logits.reshape(-1, logits.size(-1)),
            #                  targets.reshape(-1))

            # TODO: Backward pass with gradient clipping
            # optimizer.zero_grad()
            # loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), clip)
            # optimizer.step()

            # epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if epoch % print_every == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
            # Generate a sample to monitor progress
            if dataset is not None:
                sample = generate_text(model, dataset, device,
                                       seed_text="The ", length=100,
                                       temperature=0.8)
                print(f"  Sample: \"{sample}\"")

    return losses


# ============================================================
# Task 4: Text Generation
# ============================================================

def generate_text(model, dataset, device, seed_text="The ",
                  length=200, temperature=0.8):
    """Generate text by sampling from the trained model.

    Parameters
    ----------
    model : CharLSTM
        A trained character-level LSTM.
    dataset : CharDataset
        Used for character encoding/decoding.
    device : torch.device
    seed_text : str
        Initial text to prime the model.
    length : int
        Number of characters to generate.
    temperature : float
        Sampling temperature. Lower = more conservative,
        higher = more creative. Must be > 0.

    Returns
    -------
    generated : str
        The seed text followed by generated characters.
    """
    model.eval()
    chars = list(seed_text)
    hidden = model.init_hidden(1, device)

    # TODO: Feed the seed text through the model to build up hidden state
    # for ch in seed_text:
    #     if ch not in dataset.char_to_idx:
    #         continue
    #     x = torch.tensor([[dataset.char_to_idx[ch]]], device=device)
    #     logits, hidden = model(x, hidden)

    # TODO: Generate new characters one at a time
    # for _ in range(length):
    #     # Get the logits for the last character
    #     logits_last = logits[0, -1, :] / temperature
    #
    #     # Convert to probabilities
    #     probs = torch.softmax(logits_last, dim=0)
    #
    #     # Sample from the distribution
    #     idx = torch.multinomial(probs, 1).item()
    #
    #     # Append the predicted character
    #     chars.append(dataset.idx_to_char[idx])
    #
    #     # Feed the predicted character back as the next input
    #     x = torch.tensor([[idx]], device=device)
    #     logits, hidden = model(x, hidden)

    return ''.join(chars)


# ============================================================
# Visualization
# ============================================================

def plot_loss(losses, output_file='training_loss.png'):
    """Plot the training loss curve.

    Parameters
    ----------
    losses : list of float
        Loss per epoch.
    output_file : str
        Filename for the saved plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses, linewidth=2)
    plt.title('LSTM Training Loss', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Loss plot saved to {output_file}")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # --- Task 1: Prepare data ---
    print("=" * 60)
    print("Task 1: Prepare the Text Data")
    print("=" * 60)

    # You can replace SAMPLE_TEXT with your own text file:
    #   with open('my_corpus.txt', 'r') as f:
    #       text = f.read()
    text = SAMPLE_TEXT

    seq_length = 64
    batch_size = 32

    dataset = CharDataset(text, seq_length=seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True)

    print(f"  Sequence length: {seq_length}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {len(dataloader)}")

    # --- Task 2: Build model ---
    print("\n" + "=" * 60)
    print("Task 2: Build the LSTM Model")
    print("=" * 60)

    # TODO: Create the model with appropriate hyperparameters
    # model = CharLSTM(
    #     vocab_size=dataset.vocab_size,
    #     embed_size=64,
    #     hidden_size=256,
    #     num_layers=2,
    #     dropout=0.2,
    # ).to(device)
    model = CharLSTM(vocab_size=dataset.vocab_size).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    print(model)

    # --- Task 3: Train ---
    print("\n" + "=" * 60)
    print("Task 3: Train the Model")
    print("=" * 60)

    losses = train_model(
        model, dataloader, device,
        epochs=50,
        lr=0.002,
        clip=1.0,
        print_every=5,
        dataset=dataset,
    )

    # --- Task 4: Generate text ---
    print("\n" + "=" * 60)
    print("Task 4: Generate Text")
    print("=" * 60)

    seed = "The king "
    temperatures = [0.2, 0.5, 0.8, 1.2]

    for temp in temperatures:
        print(f"\n--- Temperature: {temp} ---")
        generated = generate_text(model, dataset, device,
                                  seed_text=seed, length=200,
                                  temperature=temp)
        print(generated)

    # --- Task 5: Plot loss ---
    print("\n" + "=" * 60)
    print("Task 5: Analyze Training")
    print("=" * 60)

    if any(l > 0 for l in losses):
        plot_loss(losses)
    else:
        print("  No loss recorded -- complete Task 3 to see the loss curve.")

    print("\nLab 4.2 Complete!")
    print("Key takeaways:")
    print("  - LSTMs learn sequential patterns by maintaining hidden state")
    print("  - Character-level models learn spelling, grammar, and style")
    print("  - Temperature controls the trade-off between coherence and diversity")
    print("  - Gradient clipping is essential for stable RNN/LSTM training")
