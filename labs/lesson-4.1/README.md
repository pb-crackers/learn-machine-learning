# Lab 4.1: Train Word2Vec and Explore Word Embeddings

## Objective

Train a Word2Vec model on a real text corpus, explore semantic relationships between words, perform word analogies, and visualize the embedding space using t-SNE.

## Learning Goals

- Understand how Word2Vec learns dense vector representations of words
- Train a Word2Vec model using Gensim on a real corpus
- Query the model for similar words and perform analogy tasks
- Reduce high-dimensional embeddings to 2D with t-SNE for visualization
- Interpret clusters and relationships in the embedding space

## Prerequisites

- Python 3.8+
- gensim
- scikit-learn
- matplotlib
- numpy

Install dependencies:

```bash
pip install gensim scikit-learn matplotlib numpy
```

## Dataset

This lab uses the `text8` corpus from Gensim's built-in downloader. This is a cleaned subset of Wikipedia (~100 MB of text). If the download is slow, the starter code also includes an option to use a smaller built-in dataset.

## Tasks

### Task 1: Load and Prepare the Corpus
- Download the `text8` corpus using `gensim.downloader`
- Inspect the data: count total words, unique words, and a few sample sentences
- Understand the input format that Word2Vec expects (list of lists of tokens)

### Task 2: Train a Word2Vec Model
- Train a Word2Vec model with the Skip-gram architecture (`sg=1`)
- Configure hyperparameters: vector size, window, min count, workers
- Print vocabulary size and inspect a sample word vector

### Task 3: Explore Word Similarities
- Find the top-10 most similar words to several query words (e.g., "king", "computer", "france")
- Compute cosine similarity between specific word pairs
- Identify which word does not belong in a group using `doesnt_match()`

### Task 4: Word Analogies
- Solve analogy tasks: "king - man + woman = ?"
- Try several analogies across different domains (geography, grammar, etc.)
- Discuss which analogies work well and which fail

### Task 5: Visualize with t-SNE
- Select a subset of interesting words (100-200 words from different categories)
- Extract their vectors and reduce to 2D with t-SNE
- Plot the 2D embeddings, coloring points by semantic category
- Identify and annotate clusters

## Expected Output

```
Corpus: 17,005,207 total words
Vocabulary: 71,290 words (after min_count=5)

Most similar to 'king':
  queen: 0.7019
  prince: 0.6654
  monarch: 0.6502
  throne: 0.6320
  ...

Analogy: king - man + woman = queen (0.7118)
Analogy: paris - france + germany = berlin (0.6734)
Analogy: walking - walk + swim = swimming (0.6218)

Odd one out in ['breakfast', 'lunch', 'dinner', 'python']: python

t-SNE plot saved to embeddings_tsne.png
```

(Exact numbers will vary due to random initialization.)

## Stretch Goals

- Compare Skip-gram vs. CBOW by training both and checking analogy accuracy
- Experiment with different vector sizes (50, 100, 300) and compare quality
- Load a pretrained model (`word2vec-google-news-300`) and compare results
- Compute analogy accuracy on the standard Google analogy test set
