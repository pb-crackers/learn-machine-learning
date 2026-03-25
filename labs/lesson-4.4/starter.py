"""
Lab 4.4: Fine-Tuning GPT-2 & Experimenting with Generation
===========================================================
Fine-tune a small language model and explore generation parameters.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


# ============================================================
# Exercise 1: Load & Explore GPT-2
# ============================================================

print("=" * 50)
print("Exercise 1: Load & Explore GPT-2")
print("=" * 50)

model_name = "gpt2"  # 124M parameters — small enough to fine-tune locally
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Vocabulary size: {tokenizer.vocab_size}")

# Generate text with the base model
prompt = "Machine learning is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

print(f"\nPrompt: {prompt}")
print(f"Generated: {tokenizer.decode(output[0], skip_special_tokens=True)}")
print()


# ============================================================
# Exercise 2: Prepare a Custom Dataset
# ============================================================

print("=" * 50)
print("Exercise 2: Prepare Custom Dataset")
print("=" * 50)

# TODO: Create a text file with your training data
# Option 1: Download Shakespeare
#   !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O train.txt
# Option 2: Write your own small dataset

# For this example, we'll create a small sample dataset
sample_text = """
Machine learning is the study of algorithms that improve through experience.
Neural networks are inspired by the structure of biological brains.
Deep learning uses multiple layers to learn hierarchical representations.
Transformers revolutionized natural language processing in 2017.
Attention mechanisms allow models to focus on relevant parts of the input.
"""

with open("train.txt", "w") as f:
    f.write(sample_text * 100)  # Repeat to have enough training data

print("Created train.txt with sample data")
print("(Replace this with your own dataset for better results!)")
print()

# Load the dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="train.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # GPT-2 uses causal LM, not masked LM
)

print(f"Dataset size: {len(dataset)} blocks")
print()


# ============================================================
# Exercise 3: Fine-Tune GPT-2
# ============================================================

print("=" * 50)
print("Exercise 3: Fine-Tune GPT-2")
print("=" * 50)

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    learning_rate=5e-5,
    warmup_steps=100,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# TODO: Uncomment the next line to start training
# trainer.train()
print("Uncomment trainer.train() to start fine-tuning")
print("(Training takes a few minutes on CPU, seconds on GPU)")
print()


# ============================================================
# Exercise 4: Experiment with Generation Parameters
# ============================================================

print("=" * 50)
print("Exercise 4: Generation Parameters")
print("=" * 50)


def generate_text(prompt, temperature=1.0, top_k=0, top_p=1.0, max_length=60):
    """Generate text with specified parameters."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


prompt = "The future of artificial intelligence"

# Temperature experiments
print("--- Temperature Experiments ---")
for temp in [0.3, 0.7, 1.0, 1.5]:
    result = generate_text(prompt, temperature=temp)
    print(f"\nTemperature={temp}:")
    print(f"  {result}")

# Top-k experiments
print("\n--- Top-k Experiments ---")
for k in [10, 50, 100]:
    result = generate_text(prompt, top_k=k)
    print(f"\nTop-k={k}:")
    print(f"  {result}")

# Top-p experiments
print("\n--- Top-p (nucleus) Experiments ---")
for p in [0.5, 0.9, 0.95]:
    result = generate_text(prompt, top_p=p)
    print(f"\nTop-p={p}:")
    print(f"  {result}")

print("\n\nDone! Compare the outputs above to understand each parameter's effect.")
print("Lower temperature = more deterministic, higher = more creative/random")
print("Lower top-k/top-p = more focused, higher = more diverse")
