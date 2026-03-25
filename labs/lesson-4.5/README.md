# Lab 4.5: Build a Simple RAG Pipeline

## Objective

Build a complete Retrieval-Augmented Generation (RAG) pipeline from scratch: embed documents, store them in a simple vector store, retrieve relevant context for user queries, and generate grounded answers. This is the architecture behind most production AI applications.

## Learning Goals

- Compute document embeddings using a sentence transformer model
- Build an in-memory vector store with cosine similarity search
- Implement document chunking strategies for long texts
- Retrieve relevant context for a user query
- Construct augmented prompts that ground LLM responses in retrieved documents
- Evaluate retrieval quality and understand failure modes
- Understand when and why RAG outperforms plain LLM generation

## Prerequisites

- Python 3.8+
- NumPy
- sentence-transformers

```bash
pip install numpy sentence-transformers scikit-learn
```

For the full experience with LLM generation, you will also need ONE of:
- `pip install openai` (for OpenAI API)
- `pip install anthropic` (for Claude API)
- A local model via Ollama (`brew install ollama && ollama pull llama3`)

The starter code works without an LLM API -- it builds the retrieval pipeline and constructs the augmented prompt, which you can manually paste into any LLM.

## Dataset

The lab includes a built-in knowledge base about a fictional company ("Acme Robotics") with documentation about products, policies, and technical specs. This simulates a real-world RAG use case: a customer support chatbot that needs to answer questions from company docs.

## Tasks

### Exercise 1: Document Embedding

1. Load the sample knowledge base documents
2. Initialize a sentence transformer model (`all-MiniLM-L6-v2`)
3. Compute embeddings for all documents
4. Verify embedding shapes and inspect similarity between related documents

### Exercise 2: Simple Vector Store

Build a `SimpleVectorStore` class that supports:
1. `add(documents, embeddings)` -- store documents and their vectors
2. `search(query_embedding, top_k)` -- find the k most similar documents
3. Use cosine similarity as the distance metric
4. Return documents along with their similarity scores

### Exercise 3: Document Chunking

1. Implement a chunking function that splits long documents into overlapping chunks
2. Configure chunk size (e.g., 200 tokens) and overlap (e.g., 50 tokens)
3. Re-embed the chunks and add them to the vector store
4. Compare retrieval quality with and without chunking

### Exercise 4: RAG Pipeline

Build the complete pipeline:
1. Accept a user question
2. Embed the question using the same model
3. Retrieve the top-k most relevant chunks
4. Construct an augmented prompt with the retrieved context
5. (Optional) Send to an LLM API and return the generated answer
6. Test with several questions and evaluate answer quality

### Exercise 5: Evaluation & Improvements

1. Test with questions that SHOULD be answerable from the knowledge base
2. Test with questions that SHOULD NOT be answerable (out-of-scope)
3. Implement hybrid search: combine vector similarity with keyword matching
4. Add metadata filtering (e.g., only search product docs vs. policy docs)

## Expected Output

```
Indexed 12 documents (45 chunks)
Embedding model: all-MiniLM-L6-v2 (384 dimensions)

Query: "What is the warranty period for the RoboArm Pro?"
Retrieved:
  [0.847] "The RoboArm Pro comes with a 2-year limited warranty..."
  [0.721] "All Acme Robotics products include standard warranty..."
  [0.683] "The RoboArm Pro is our flagship industrial robotic arm..."

Augmented prompt constructed (1,247 tokens)

--- RAG Answer ---
"The RoboArm Pro comes with a 2-year limited warranty that covers
manufacturing defects and component failures under normal use..."

--- Without RAG ---
"I don't have specific information about the RoboArm Pro warranty..."
```

## Stretch Goals

- Replace the in-memory store with ChromaDB (`pip install chromadb`)
- Implement re-ranking: use a cross-encoder to re-score retrieved documents
- Add query expansion: generate multiple query variants for better recall
- Build a simple chat interface that maintains conversation history
- Measure retrieval metrics: precision@k, recall, mean reciprocal rank
- Compare different embedding models and measure retrieval quality differences

## Starter Code

See `starter.py` for the skeleton implementation with all exercises structured.
