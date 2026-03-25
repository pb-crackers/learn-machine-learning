# Lab 4.4: Fine-Tuning GPT-2 & Experimenting with Generation

## Objective

Fine-tune a small GPT-2 model (124M parameters) on a custom text dataset using Hugging Face Transformers, then systematically experiment with different generation parameters (temperature, top-k, top-p) to understand how they affect output quality, diversity, and coherence.

## Learning Goals

- Load and interact with a pre-trained GPT-2 model
- Understand the tokenization process (BPE) and how text becomes token IDs
- Prepare a custom text dataset for causal language model fine-tuning
- Fine-tune GPT-2 using Hugging Face's Trainer API
- Experiment with temperature, top-k, and top-p sampling and understand their effects
- Analyze the trade-off between coherence and creativity in text generation

## Prerequisites

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- Datasets library

```bash
pip install torch transformers datasets
```

A GPU is recommended but not required -- fine-tuning GPT-2 (small) is feasible on CPU in a few minutes with a small dataset.

## Dataset Options

Choose one of these for fine-tuning (or bring your own):

1. **TinyShakespeare** (recommended for quick experiments):
   ```bash
   wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O train.txt
   ```

2. **Custom domain text**: Collect text from a specific domain (recipes, legal documents, code comments, etc.) and save it to `train.txt`.

3. **The starter code provides a small built-in sample** for testing the pipeline.

## Tasks

### Exercise 1: Load & Explore GPT-2

1. Load the pre-trained GPT-2 model and tokenizer
2. Print the model architecture and parameter count
3. Explore tokenization: encode and decode sample text, inspect token IDs
4. Generate text with the base model to see its default behavior

### Exercise 2: Prepare a Custom Dataset

1. Create or download a text file for training
2. Load it using Hugging Face's `TextDataset` (or a custom Dataset)
3. Set up a `DataCollatorForLanguageModeling` with `mlm=False` (causal LM)
4. Inspect the tokenized dataset: check sequence lengths, total tokens

### Exercise 3: Fine-Tune GPT-2

1. Configure `TrainingArguments` with appropriate hyperparameters
2. Create a `Trainer` and run training
3. Monitor training loss -- it should decrease steadily
4. Generate text from the fine-tuned model and compare with the base model

### Exercise 4: Experiment with Generation Parameters

Systematically test different generation settings:

| Parameter | Values to Try | What It Controls |
|-----------|--------------|-----------------|
| `temperature` | 0.3, 0.7, 1.0, 1.5 | Sharpness of probability distribution |
| `top_k` | 10, 50, 100 | Number of tokens considered |
| `top_p` | 0.5, 0.9, 0.95 | Cumulative probability threshold |

For each setting:
1. Generate 3 samples with the same prompt
2. Rate each on coherence (1-5) and creativity (1-5)
3. Note any repetition, nonsensical output, or interesting patterns

### Exercise 5: Analysis

1. Create a table comparing generation quality across parameter settings
2. Identify the best settings for factual vs. creative text
3. Test what happens with extreme settings (temp=0.01, temp=3.0)
4. Plot training loss curve

## Expected Output

```
Model parameters: 124,439,808
Vocabulary size: 50,257

--- Base Model ---
Prompt: "Machine learning is"
Generated: "Machine learning is a technique that allows you to..."

--- Fine-Tuned Model ---
Prompt: "Machine learning is"
Generated: (text in the style of your training data)

--- Temperature Experiments ---
Temperature=0.3: (focused, repetitive)
Temperature=0.7: (balanced, coherent)
Temperature=1.0: (diverse, occasional errors)
Temperature=1.5: (very creative, often incoherent)
```

## Stretch Goals

- Try LoRA fine-tuning using the `peft` library and compare training speed and quality
- Fine-tune on two different datasets and swap between the styles at inference
- Implement beam search decoding manually and compare with sampling
- Measure perplexity on held-out text before and after fine-tuning
- Experiment with repetition penalty and length penalty parameters

## Starter Code

See `starter.py` for the skeleton implementation with all exercises structured.
