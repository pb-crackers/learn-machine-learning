"""
Lab 4.1: Train Word2Vec and Explore Word Embeddings
====================================================

Train a Word2Vec model on a text corpus, explore semantic
relationships, perform word analogies, and visualize the
embedding space with t-SNE.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import gensim.downloader as api
from gensim.models import Word2Vec
from sklearn.manifold import TSNE


# ============================================================
# Task 1: Load and Prepare the Corpus
# ============================================================

def load_corpus(dataset_name='text8'):
    """Download and load a text corpus from Gensim's data repository.

    Parameters
    ----------
    dataset_name : str
        Name of the corpus. Use 'text8' for a ~100 MB Wikipedia
        subset, or '__testing_multiclass-classification' for a
        smaller test dataset.

    Returns
    -------
    corpus : iterable of list of str
        Each element is a list of tokens (one sentence/chunk).
    """
    print(f"Downloading '{dataset_name}' corpus (this may take a minute)...")
    corpus = api.load(dataset_name)

    # text8 returns a single iterable of sentences (lists of words).
    # We need to materialize it so we can iterate multiple times.
    # For large corpora, keep only the first N chunks to speed things up.
    sentences = list(corpus)

    return sentences


def inspect_corpus(sentences, n_preview=3):
    """Print basic statistics about the loaded corpus.

    Parameters
    ----------
    sentences : list of list of str
        The tokenized corpus.
    n_preview : int
        Number of sentences to preview.
    """
    total_words = sum(len(s) for s in sentences)
    all_words = [w for s in sentences for w in s]
    unique_words = len(set(all_words))

    print(f"\nCorpus Statistics:")
    print(f"  Number of chunks/sentences: {len(sentences)}")
    print(f"  Total words: {total_words:,}")
    print(f"  Unique words: {unique_words:,}")
    print(f"\nFirst {n_preview} sentence previews (first 15 words each):")
    for i, s in enumerate(sentences[:n_preview]):
        print(f"  [{i}] {' '.join(s[:15])}...")


# ============================================================
# Task 2: Train a Word2Vec Model
# ============================================================

def train_word2vec(sentences, vector_size=100, window=5, min_count=5,
                   sg=1, workers=4, epochs=5):
    """Train a Word2Vec model on the given corpus.

    Parameters
    ----------
    sentences : list of list of str
        Tokenized corpus.
    vector_size : int
        Dimensionality of the word vectors.
    window : int
        Maximum distance between the current and predicted word.
    min_count : int
        Ignore words that appear fewer than this many times.
    sg : int
        1 for Skip-gram, 0 for CBOW.
    workers : int
        Number of CPU threads for training.
    epochs : int
        Number of training passes over the corpus.

    Returns
    -------
    model : Word2Vec
        The trained Word2Vec model.
    """
    print(f"\nTraining Word2Vec (sg={sg}, dim={vector_size}, "
          f"window={window}, min_count={min_count})...")

    # TODO: Create and train the Word2Vec model
    # model = Word2Vec(
    #     sentences=sentences,
    #     vector_size=vector_size,
    #     window=window,
    #     min_count=min_count,
    #     sg=sg,
    #     workers=workers,
    #     epochs=epochs,
    # )
    model = None  # Replace with the code above

    # TODO: Print vocabulary size and vector shape
    # print(f"  Vocabulary size: {len(model.wv):,}")
    # print(f"  Vector dimensions: {model.wv.vector_size}")
    #
    # # Show an example vector
    # example_word = 'king'
    # if example_word in model.wv:
    #     vec = model.wv[example_word]
    #     print(f"\n  Vector for '{example_word}' (first 10 dims):")
    #     print(f"  {vec[:10]}")
    #     print(f"  Vector norm: {np.linalg.norm(vec):.4f}")

    return model


# ============================================================
# Task 3: Explore Word Similarities
# ============================================================

def explore_similarities(model):
    """Find similar words and compute pairwise similarities.

    Parameters
    ----------
    model : Word2Vec
        A trained Word2Vec model.
    """
    wv = model.wv

    # --- Most Similar Words ---
    query_words = ['king', 'computer', 'france', 'dog', 'music']
    print("\n" + "=" * 50)
    print("Most Similar Words")
    print("=" * 50)

    for word in query_words:
        if word not in wv:
            print(f"\n  '{word}' not in vocabulary, skipping.")
            continue
        # TODO: Find top 10 most similar words
        # similar = wv.most_similar(word, topn=10)
        # print(f"\n  '{word}':")
        # for w, score in similar:
        #     print(f"    {w:20s} {score:.4f}")
        pass

    # --- Pairwise Similarity ---
    print("\n" + "=" * 50)
    print("Pairwise Similarities")
    print("=" * 50)

    word_pairs = [
        ('king', 'queen'),
        ('king', 'car'),
        ('dog', 'cat'),
        ('dog', 'building'),
        ('france', 'germany'),
    ]

    for w1, w2 in word_pairs:
        if w1 in wv and w2 in wv:
            # TODO: Compute cosine similarity between word pairs
            # sim = wv.similarity(w1, w2)
            # print(f"  sim('{w1}', '{w2}') = {sim:.4f}")
            pass

    # --- Odd One Out ---
    print("\n" + "=" * 50)
    print("Odd One Out")
    print("=" * 50)

    word_groups = [
        ['breakfast', 'lunch', 'dinner', 'python'],
        ['cat', 'dog', 'horse', 'television'],
        ['france', 'germany', 'italy', 'banana'],
    ]

    for group in word_groups:
        valid = [w for w in group if w in wv]
        if len(valid) == len(group):
            # TODO: Find the word that doesn't match
            # outlier = wv.doesnt_match(group)
            # print(f"  {group} -> '{outlier}'")
            pass
        else:
            print(f"  Skipping {group} (words missing from vocabulary)")


# ============================================================
# Task 4: Word Analogies
# ============================================================

def test_analogies(model):
    """Test the model on word analogy tasks.

    The analogy "a is to b as c is to ?" is solved by finding
    the word closest to: vec(b) - vec(a) + vec(c).

    Parameters
    ----------
    model : Word2Vec
        A trained Word2Vec model.
    """
    wv = model.wv

    print("\n" + "=" * 50)
    print("Word Analogies (a : b :: c : ?)")
    print("=" * 50)

    # Format: (a, b, c, expected_answer)
    analogies = [
        ('man', 'king', 'woman', 'queen'),
        ('france', 'paris', 'germany', 'berlin'),
        ('walk', 'walking', 'swim', 'swimming'),
        ('man', 'doctor', 'woman', 'nurse'),
        ('small', 'smaller', 'big', 'bigger'),
        ('japan', 'tokyo', 'france', 'paris'),
        ('good', 'better', 'bad', 'worse'),
    ]

    correct = 0
    total = 0

    for a, b, c, expected in analogies:
        # Skip if any word is missing from vocabulary
        if not all(w in wv for w in [a, b, c]):
            print(f"\n  {a} : {b} :: {c} : ? -> [skipped, word not in vocab]")
            continue

        # TODO: Solve the analogy using most_similar with
        #       positive=[b, c] and negative=[a]
        # result = wv.most_similar(positive=[b, c], negative=[a], topn=5)
        # predicted = result[0][0]
        # score = result[0][1]
        #
        # is_correct = predicted == expected
        # if is_correct:
        #     correct += 1
        # total += 1
        #
        # mark = "OK" if is_correct else "XX"
        # print(f"\n  {a} : {b} :: {c} : ?")
        # print(f"    Expected: {expected}")
        # print(f"    Got:      {predicted} ({score:.4f}) [{mark}]")
        # print(f"    Top 5:    {[w for w, s in result]}")
        pass

    # if total > 0:
    #     print(f"\n  Analogy accuracy: {correct}/{total} "
    #           f"({100*correct/total:.0f}%)")


# ============================================================
# Task 5: Visualize with t-SNE
# ============================================================

def visualize_embeddings(model, output_file='embeddings_tsne.png'):
    """Visualize word embeddings in 2D using t-SNE.

    Parameters
    ----------
    model : Word2Vec
        A trained Word2Vec model.
    output_file : str
        Filename for the saved plot.
    """
    wv = model.wv

    # Define word categories to visualize
    categories = {
        'Countries': [
            'france', 'germany', 'italy', 'spain', 'japan',
            'china', 'india', 'brazil', 'russia', 'canada',
            'australia', 'mexico', 'egypt', 'sweden', 'portugal',
        ],
        'Animals': [
            'dog', 'cat', 'horse', 'fish', 'bird',
            'bear', 'lion', 'tiger', 'elephant', 'wolf',
            'rabbit', 'snake', 'monkey', 'deer', 'eagle',
        ],
        'Technology': [
            'computer', 'software', 'internet', 'algorithm', 'database',
            'network', 'server', 'programming', 'processor', 'digital',
            'memory', 'system', 'data', 'technology', 'electronic',
        ],
        'Science': [
            'physics', 'chemistry', 'biology', 'mathematics', 'theory',
            'experiment', 'research', 'molecule', 'energy', 'quantum',
            'evolution', 'cell', 'atom', 'gravity', 'particle',
        ],
        'Music': [
            'music', 'song', 'album', 'band', 'guitar',
            'piano', 'orchestra', 'rhythm', 'melody', 'jazz',
            'rock', 'concert', 'singer', 'composer', 'symphony',
        ],
    }

    # Collect words that exist in the vocabulary
    words = []
    labels = []
    colors_list = []
    color_map = {
        'Countries': '#e41a1c',
        'Animals': '#377eb8',
        'Technology': '#4daf4a',
        'Science': '#984ea3',
        'Music': '#ff7f00',
    }

    for category, word_list in categories.items():
        for word in word_list:
            if word in wv:
                words.append(word)
                labels.append(category)
                colors_list.append(color_map[category])

    if len(words) < 10:
        print("Not enough words found in vocabulary for visualization.")
        return

    print(f"\nVisualizing {len(words)} words from {len(categories)} categories...")

    # TODO: Extract word vectors into a matrix
    # vectors = np.array([wv[w] for w in words])

    # TODO: Run t-SNE to reduce to 2 dimensions
    # tsne = TSNE(n_components=2, random_state=42, perplexity=30,
    #             n_iter=1000)
    # coords = tsne.fit_transform(vectors)

    # TODO: Create the scatter plot
    # plt.figure(figsize=(16, 12))
    #
    # # Plot each category with its color
    # for category in categories:
    #     mask = [l == category for l in labels]
    #     indices = [i for i, m in enumerate(mask) if m]
    #     if indices:
    #         cat_coords = coords[indices]
    #         plt.scatter(cat_coords[:, 0], cat_coords[:, 1],
    #                     c=color_map[category], label=category,
    #                     s=80, alpha=0.7, edgecolors='white', linewidths=0.5)
    #
    # # Annotate each point with its word
    # for i, word in enumerate(words):
    #     plt.annotate(word, (coords[i, 0], coords[i, 1]),
    #                  fontsize=8, alpha=0.8,
    #                  xytext=(5, 5), textcoords='offset points')
    #
    # plt.title('Word2Vec Embeddings Visualized with t-SNE', fontsize=16)
    # plt.xlabel('t-SNE dimension 1')
    # plt.ylabel('t-SNE dimension 2')
    # plt.legend(fontsize=12, loc='best')
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(output_file, dpi=150, bbox_inches='tight')
    # plt.show()
    # print(f"  Plot saved to {output_file}")

    pass


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    # Task 1: Load corpus
    print("=" * 60)
    print("Task 1: Load and Inspect the Corpus")
    print("=" * 60)
    sentences = load_corpus('text8')
    inspect_corpus(sentences)

    # Task 2: Train Word2Vec
    print("\n" + "=" * 60)
    print("Task 2: Train Word2Vec")
    print("=" * 60)
    model = train_word2vec(
        sentences,
        vector_size=100,
        window=5,
        min_count=5,
        sg=1,          # Skip-gram
        workers=4,
        epochs=5,
    )

    if model is not None:
        # Task 3: Explore similarities
        print("\n" + "=" * 60)
        print("Task 3: Word Similarities")
        print("=" * 60)
        explore_similarities(model)

        # Task 4: Word analogies
        print("\n" + "=" * 60)
        print("Task 4: Word Analogies")
        print("=" * 60)
        test_analogies(model)

        # Task 5: t-SNE visualization
        print("\n" + "=" * 60)
        print("Task 5: t-SNE Visualization")
        print("=" * 60)
        visualize_embeddings(model)
    else:
        print("\nModel is None -- complete Task 2 to proceed.")

    print("\nLab 4.1 Complete!")
    print("Key takeaways:")
    print("  - Word2Vec captures semantic relationships as vector arithmetic")
    print("  - Similar words cluster together in embedding space")
    print("  - Analogies work because relationships are encoded as offsets")
    print("  - t-SNE reveals thematic clusters in the learned representations")
