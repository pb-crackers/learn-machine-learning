"""
Lab 4.5: Build a Simple RAG Pipeline
====================================
Retrieval-Augmented Generation from scratch.
"""

import numpy as np


# ============================================================
# Exercise 1: Document Embedding
# ============================================================

print("=" * 50)
print("Exercise 1: Document Embedding")
print("=" * 50)

# Sample knowledge base about machine learning
documents = [
    "Linear regression finds the best-fitting line through data by minimizing the sum of squared errors between predicted and actual values.",
    "Decision trees split data based on feature values, creating a tree structure where each leaf represents a prediction.",
    "Neural networks consist of layers of interconnected neurons that learn to transform inputs into outputs through training.",
    "Gradient descent is an optimization algorithm that iteratively adjusts model parameters by moving in the direction of steepest decrease in the loss function.",
    "Overfitting occurs when a model learns the training data too well, including its noise, and fails to generalize to new data.",
    "Cross-validation splits data into multiple folds and trains on different combinations to get a robust estimate of model performance.",
    "Transformers use self-attention mechanisms to process all tokens in a sequence simultaneously, enabling parallel computation.",
    "Word embeddings represent words as dense vectors where similar words are close together in the vector space.",
    "Convolutional neural networks use filters that slide over input data to detect local patterns like edges and textures.",
    "Regularization techniques like L1, L2, and dropout help prevent overfitting by constraining model complexity.",
    "Batch normalization normalizes the inputs of each layer, which helps stabilize and speed up training.",
    "Transfer learning uses a model pretrained on a large dataset as a starting point for a new task with limited data.",
]

# Option A: Use sentence-transformers (recommended)
try:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")  # Small, fast model
    embeddings = model.encode(documents)
    print(f"Using sentence-transformers model")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    def embed_text(text):
        return model.encode([text])[0]

except ImportError:
    # Option B: Simple TF-IDF-based embeddings as fallback
    print("sentence-transformers not installed, using simple TF-IDF fallback")
    print("Install with: pip install sentence-transformers")

    from collections import Counter
    import re

    def tokenize(text):
        return re.findall(r'\w+', text.lower())

    # Build vocabulary
    all_tokens = set()
    for doc in documents:
        all_tokens.update(tokenize(doc))
    vocab = sorted(all_tokens)
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    def simple_embed(text):
        tokens = tokenize(text)
        counts = Counter(tokens)
        vec = np.zeros(len(vocab))
        for token, count in counts.items():
            if token in word_to_idx:
                vec[word_to_idx[token]] = count
        # Normalize
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    embeddings = np.array([simple_embed(doc) for doc in documents])
    embed_text = simple_embed
    print(f"Embedding dimension: {embeddings.shape[1]}")

print(f"Embedded {len(documents)} documents")
print(f"Embeddings shape: {embeddings.shape}")
print()


# ============================================================
# Exercise 2: Simple Vector Store
# ============================================================

print("=" * 50)
print("Exercise 2: Simple Vector Store")
print("=" * 50)


class SimpleVectorStore:
    """A minimal in-memory vector store with cosine similarity search."""

    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add_documents(self, docs, embeds):
        """Add documents and their embeddings to the store."""
        self.documents.extend(docs)
        self.embeddings = list(embeds) if not self.embeddings else self.embeddings + list(embeds)

    def cosine_similarity(self, a, b):
        """Compute cosine similarity between two vectors."""
        # TODO: Implement cosine similarity
        # cos_sim = (a · b) / (||a|| * ||b||)
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def search(self, query_embedding, top_k=3):
        """
        Find the top-k most similar documents to the query.

        Returns: List of (document, score) tuples
        """
        scores = []
        for i, emb in enumerate(self.embeddings):
            sim = self.cosine_similarity(query_embedding, emb)
            scores.append((i, sim))

        # Sort by similarity (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            results.append((self.documents[idx], score))

        return results


# Create and populate the vector store
store = SimpleVectorStore()
store.add_documents(documents, embeddings)
print(f"Vector store contains {len(store.documents)} documents")
print()


# ============================================================
# Exercise 3: Retrieval
# ============================================================

print("=" * 50)
print("Exercise 3: Retrieval")
print("=" * 50)

queries = [
    "How do neural networks learn?",
    "What is overfitting and how to prevent it?",
    "How do transformers work?",
]

for query in queries:
    print(f"\nQuery: {query}")
    query_emb = embed_text(query)
    results = store.search(query_emb, top_k=3)

    for i, (doc, score) in enumerate(results):
        print(f"  {i+1}. [{score:.3f}] {doc[:80]}...")


# ============================================================
# Exercise 4: RAG Pipeline
# ============================================================

print("\n" + "=" * 50)
print("Exercise 4: RAG Pipeline")
print("=" * 50)


def rag_answer(query, store, top_k=3):
    """
    Simple RAG pipeline:
    1. Embed the query
    2. Retrieve relevant documents
    3. Format a prompt with context
    4. (In production, send to an LLM)
    """
    # Step 1: Embed the query
    query_emb = embed_text(query)

    # Step 2: Retrieve relevant context
    results = store.search(query_emb, top_k=top_k)

    # Step 3: Format the prompt
    context = "\n".join([f"- {doc}" for doc, _ in results])

    prompt = f"""Answer the question based on the following context.

Context:
{context}

Question: {query}

Answer:"""

    print(f"\n--- Generated Prompt ---")
    print(prompt)
    print(f"--- End Prompt ---\n")

    # Step 4: In production, you would send this to an LLM:
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # return response.choices[0].message.content

    print("(In production, this prompt would be sent to an LLM for a natural language answer)")
    return context


# Run the RAG pipeline
queries = [
    "What technique helps prevent a model from memorizing training data?",
    "How do transformers process sequences differently from RNNs?",
]

for query in queries:
    print(f"\n{'='*40}")
    print(f"User Question: {query}")
    rag_answer(query, store)


print("\n\nDone! You've built a complete RAG pipeline.")
print("Next steps:")
print("  1. Replace the sample docs with a real knowledge base")
print("  2. Use sentence-transformers for better embeddings")
print("  3. Connect to an LLM API for generation")
print("  4. Try a real vector database (e.g., ChromaDB, FAISS)")
