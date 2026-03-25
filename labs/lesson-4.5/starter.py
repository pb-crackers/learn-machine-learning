"""
Lab 4.5: Build a Simple RAG Pipeline
=====================================

Build a Retrieval-Augmented Generation pipeline: embed documents,
store them in a vector store, retrieve relevant context, and
generate grounded answers.
"""

import numpy as np
from typing import List, Tuple, Optional

# Attempt to import sentence-transformers; provide fallback guidance
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    print("WARNING: sentence-transformers not installed.")
    print("Install with: pip install sentence-transformers")
    print("Falling back to random embeddings for demonstration.\n")


# ============================================================
# Sample Knowledge Base (Fictional Company: Acme Robotics)
# ============================================================

KNOWLEDGE_BASE = [
    {
        "id": "prod-001",
        "category": "product",
        "title": "RoboArm Pro",
        "content": (
            "The RoboArm Pro is Acme Robotics' flagship industrial robotic arm. "
            "It features 6 degrees of freedom, a maximum payload of 10 kg, and "
            "a reach of 1.2 meters. The arm uses brushless servo motors for "
            "precise positioning with repeatability of +/-0.02 mm. It is designed "
            "for manufacturing, assembly, and pick-and-place operations. The "
            "RoboArm Pro supports ROS 2 integration and can be programmed using "
            "Python, C++, or Acme's proprietary visual programming interface."
        ),
    },
    {
        "id": "prod-002",
        "category": "product",
        "title": "RoboArm Lite",
        "content": (
            "The RoboArm Lite is a compact robotic arm designed for education "
            "and prototyping. It has 5 degrees of freedom, a payload of 0.5 kg, "
            "and a reach of 0.4 meters. It uses stepper motors and is controlled "
            "via USB or Wi-Fi. The Lite comes with a Python SDK and tutorial "
            "curriculum for teaching robotics fundamentals. It is priced at $299 "
            "for educational institutions."
        ),
    },
    {
        "id": "prod-003",
        "category": "product",
        "title": "NavBot Autonomous Platform",
        "content": (
            "The NavBot is a wheeled autonomous mobile platform for warehouse "
            "and logistics applications. It uses LiDAR and camera-based SLAM "
            "for navigation, can carry up to 200 kg, and operates for 8 hours "
            "on a single charge. The NavBot integrates with standard warehouse "
            "management systems via REST API and supports fleet coordination "
            "for multi-robot deployments."
        ),
    },
    {
        "id": "warranty-001",
        "category": "policy",
        "title": "Standard Warranty Policy",
        "content": (
            "All Acme Robotics products include a standard warranty. The "
            "RoboArm Pro comes with a 2-year limited warranty covering "
            "manufacturing defects and component failures under normal use. "
            "The RoboArm Lite has a 1-year warranty. The NavBot platform "
            "carries a 3-year warranty on the chassis and 1-year on the "
            "battery. Extended warranty plans are available for purchase "
            "within 30 days of the original purchase date."
        ),
    },
    {
        "id": "warranty-002",
        "category": "policy",
        "title": "Warranty Exclusions",
        "content": (
            "The warranty does not cover damage caused by misuse, unauthorized "
            "modifications, exposure to extreme environments beyond rated specs, "
            "or normal wear and tear. Software issues are covered under a "
            "separate software support agreement. Consumable parts such as "
            "gripper pads and drive belts are not covered under warranty."
        ),
    },
    {
        "id": "support-001",
        "category": "support",
        "title": "Technical Support",
        "content": (
            "Acme Robotics offers three tiers of technical support. Basic "
            "support (included with purchase) provides email support with "
            "48-hour response time and access to the online knowledge base. "
            "Professional support ($500/year) adds phone support, 4-hour "
            "response time, and remote diagnostics. Enterprise support "
            "(custom pricing) includes on-site engineers, 1-hour response "
            "time, and dedicated account management."
        ),
    },
    {
        "id": "return-001",
        "category": "policy",
        "title": "Return Policy",
        "content": (
            "Products may be returned within 30 days of delivery for a full "
            "refund, provided they are in original packaging and undamaged. "
            "Custom-configured units are non-refundable. Return shipping is "
            "the responsibility of the customer unless the return is due to "
            "a defect. Refunds are processed within 10 business days of "
            "receiving the returned product."
        ),
    },
    {
        "id": "spec-001",
        "category": "technical",
        "title": "RoboArm Pro Specifications",
        "content": (
            "RoboArm Pro technical specifications: Weight: 28 kg. Power: "
            "48V DC, 500W max draw. Communication: EtherCAT, CAN bus, "
            "Ethernet (TCP/IP). Operating temperature: 0-45 degrees C. IP rating: "
            "IP54. Noise level: <65 dB. Certifications: CE, UL, ISO 10218-1. "
            "Joint speeds: J1-J3: 180 deg/s, J4-J6: 360 deg/s. Software: ROS 2 "
            "Humble, Acme SDK v3.2+, Python 3.9+."
        ),
    },
    {
        "id": "spec-002",
        "category": "technical",
        "title": "NavBot Specifications",
        "content": (
            "NavBot technical specifications: Dimensions: 800x600x350 mm. "
            "Weight: 45 kg (unloaded). Max speed: 2 m/s. LiDAR: Velodyne "
            "VLP-16 (optional upgrade to VLP-32). Cameras: 4x Intel RealSense "
            "D435i. Compute: NVIDIA Jetson Orin. Battery: 48V 30Ah LiFePO4. "
            "Charging time: 3 hours. Connectivity: Wi-Fi 6, 5G (optional), "
            "Bluetooth 5.2."
        ),
    },
    {
        "id": "setup-001",
        "category": "support",
        "title": "RoboArm Pro Setup Guide",
        "content": (
            "To set up the RoboArm Pro: 1) Mount the base to a stable surface "
            "rated for 50 kg using the provided M10 bolts. 2) Connect the 48V "
            "power supply. 3) Connect the EtherCAT or Ethernet cable to your "
            "control computer. 4) Install the Acme SDK: pip install acme-robotics. "
            "5) Run the calibration routine: acme-calibrate --model pro. "
            "6) Verify with the test script: python -m acme.test_arm. The "
            "calibration process takes approximately 5 minutes."
        ),
    },
    {
        "id": "pricing-001",
        "category": "product",
        "title": "Pricing",
        "content": (
            "Current product pricing (as of 2025): RoboArm Pro: $12,500 "
            "(base configuration). RoboArm Pro with force-torque sensor: "
            "$14,200. RoboArm Pro with vision package: $15,800. RoboArm "
            "Lite: $299 (educational), $499 (commercial license). NavBot: "
            "$35,000 (single unit), volume discounts available for fleet "
            "orders of 10+ units. All prices are USD, excluding shipping "
            "and applicable taxes."
        ),
    },
    {
        "id": "safety-001",
        "category": "technical",
        "title": "Safety Guidelines",
        "content": (
            "The RoboArm Pro must be operated within a safety enclosure "
            "compliant with ISO 10218-2 when running at full speed. "
            "Collaborative mode (reduced speed, force-limited) allows "
            "operation without enclosure but requires a risk assessment. "
            "Emergency stop buttons must be accessible within 2 meters of "
            "the operating area. All operators must complete the Acme "
            "Robotics safety training course before operating any equipment."
        ),
    },
]


# ============================================================
# Exercise 1: Document Embedding
# ============================================================

class DocumentEmbedder:
    """Embed documents using a sentence transformer model."""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the embedder.

        Parameters
        ----------
        model_name : str
            Name of the sentence-transformers model to use.
        """
        if HAS_SBERT:
            print(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        else:
            print("Using random embeddings (install sentence-transformers for real ones)")
            self.model = None
            self.embedding_dim = 384  # Mimic MiniLM dimensions

        print(f"Embedding dimension: {self.embedding_dim}")

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for a list of texts.

        Parameters
        ----------
        texts : list of str
            Texts to embed.

        Returns
        -------
        embeddings : np.ndarray, shape (len(texts), embedding_dim)
        """
        if self.model is not None:
            # TODO: Use the sentence transformer model to encode texts
            # embeddings = self.model.encode(texts, show_progress_bar=True)
            # return np.array(embeddings)
            pass

        # Fallback: random embeddings (for testing without sentence-transformers)
        np.random.seed(42)
        return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)


# ============================================================
# Exercise 2: Simple Vector Store
# ============================================================

class SimpleVectorStore:
    """An in-memory vector store with cosine similarity search."""

    def __init__(self):
        self.documents = []       # List of document dicts
        self.embeddings = None    # np.ndarray of shape (n_docs, embedding_dim)

    def add(self, documents: List[dict], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the store.

        Parameters
        ----------
        documents : list of dict
            Each dict should have at least 'content' and 'id' keys.
        embeddings : np.ndarray, shape (n_docs, embedding_dim)
        """
        # TODO: Store the documents and embeddings
        # self.documents.extend(documents)
        # if self.embeddings is None:
        #     self.embeddings = embeddings
        # else:
        #     self.embeddings = np.vstack([self.embeddings, embeddings])
        #
        # print(f"  Added {len(documents)} documents. "
        #       f"Total: {len(self.documents)}")
        pass

    def search(self, query_embedding: np.ndarray, top_k: int = 3
               ) -> List[Tuple[dict, float]]:
        """
        Find the most similar documents to the query.

        Parameters
        ----------
        query_embedding : np.ndarray, shape (embedding_dim,)
        top_k : int
            Number of results to return.

        Returns
        -------
        results : list of (document_dict, similarity_score)
            Sorted by descending similarity.
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []

        # TODO: Compute cosine similarity between query and all documents
        # 1. Normalize the query embedding
        # query_norm = query_embedding / np.linalg.norm(query_embedding)
        #
        # 2. Normalize all document embeddings
        # doc_norms = self.embeddings / np.linalg.norm(
        #     self.embeddings, axis=1, keepdims=True
        # )
        #
        # 3. Compute cosine similarity (dot product of normalized vectors)
        # similarities = np.dot(doc_norms, query_norm)
        #
        # 4. Get top-k indices
        # top_indices = np.argsort(similarities)[::-1][:top_k]
        #
        # 5. Return documents with scores
        # results = [
        #     (self.documents[i], float(similarities[i]))
        #     for i in top_indices
        # ]
        # return results

        return []  # Replace with the code above


# ============================================================
# Exercise 3: Document Chunking
# ============================================================

def chunk_document(text: str, chunk_size: int = 200,
                   overlap: int = 50) -> List[str]:
    """
    Split a document into overlapping chunks.

    Parameters
    ----------
    text : str
        The document text.
    chunk_size : int
        Maximum number of words per chunk.
    overlap : int
        Number of overlapping words between consecutive chunks.

    Returns
    -------
    chunks : list of str
    """
    # TODO: Split text into overlapping chunks
    # words = text.split()
    # chunks = []
    # start = 0
    #
    # while start < len(words):
    #     end = start + chunk_size
    #     chunk = " ".join(words[start:end])
    #     chunks.append(chunk)
    #     start = end - overlap
    #
    #     # Avoid infinite loop for very small texts
    #     if end >= len(words):
    #         break
    #
    # return chunks

    # Placeholder: return entire text as single chunk
    return [text]


def prepare_knowledge_base(kb: List[dict], chunking: bool = False,
                           chunk_size: int = 200, overlap: int = 50
                           ) -> List[dict]:
    """
    Prepare knowledge base documents, optionally chunking long ones.

    Parameters
    ----------
    kb : list of dict
        Raw knowledge base entries.
    chunking : bool
        Whether to split documents into chunks.
    chunk_size : int
        Words per chunk (if chunking).
    overlap : int
        Overlap words (if chunking).

    Returns
    -------
    processed : list of dict
        Each dict has 'id', 'content', 'title', 'category' keys.
    """
    processed = []

    for doc in kb:
        if chunking:
            chunks = chunk_document(doc["content"], chunk_size, overlap)
            for i, chunk in enumerate(chunks):
                processed.append({
                    "id": f"{doc['id']}-chunk-{i}",
                    "content": chunk,
                    "title": doc["title"],
                    "category": doc["category"],
                    "parent_id": doc["id"],
                })
        else:
            processed.append({
                "id": doc["id"],
                "content": doc["content"],
                "title": doc["title"],
                "category": doc["category"],
            })

    return processed


# ============================================================
# Exercise 4: RAG Pipeline
# ============================================================

class RAGPipeline:
    """A simple Retrieval-Augmented Generation pipeline."""

    def __init__(self, embedder: DocumentEmbedder, store: SimpleVectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int = 3
                 ) -> List[Tuple[dict, float]]:
        """
        Retrieve relevant documents for a query.

        Parameters
        ----------
        query : str
            The user's question.
        top_k : int
            Number of documents to retrieve.

        Returns
        -------
        results : list of (document_dict, similarity_score)
        """
        # TODO: Embed the query and search the vector store
        # query_embedding = self.embedder.embed([query])[0]
        # return self.store.search(query_embedding, top_k=top_k)

        return []  # Replace with the code above

    def build_prompt(self, query: str, retrieved: List[Tuple[dict, float]],
                     system_prompt: Optional[str] = None) -> str:
        """
        Build an augmented prompt with retrieved context.

        Parameters
        ----------
        query : str
            The user's question.
        retrieved : list of (document_dict, similarity_score)
            Retrieved documents and their scores.
        system_prompt : str, optional
            Custom system instructions.

        Returns
        -------
        prompt : str
            The full prompt to send to an LLM.
        """
        if system_prompt is None:
            system_prompt = (
                "You are a helpful customer support assistant for Acme Robotics. "
                "Answer the user's question based ONLY on the provided context. "
                "If the context does not contain enough information to answer, "
                "say 'I don't have enough information to answer that question.' "
                "Always cite which document your answer is based on."
            )

        # TODO: Format the retrieved documents as context
        # context_parts = []
        # for i, (doc, score) in enumerate(retrieved):
        #     context_parts.append(
        #         f"[Document: {doc['title']} (relevance: {score:.3f})]\n"
        #         f"{doc['content']}"
        #     )
        # context = "\n\n".join(context_parts)
        #
        # prompt = f"""{system_prompt}
        #
        # Context:
        # {context}
        #
        # User question: {query}
        #
        # Answer:"""
        #
        # return prompt

        return f"[Prompt construction not implemented]\nQuery: {query}"

    def query(self, question: str, top_k: int = 3) -> str:
        """
        Full RAG pipeline: retrieve + build prompt.

        Parameters
        ----------
        question : str
        top_k : int

        Returns
        -------
        prompt : str
            The augmented prompt (ready to send to an LLM).
        """
        print(f"\nQuery: \"{question}\"")

        # Retrieve
        retrieved = self.retrieve(question, top_k=top_k)

        if retrieved:
            print(f"\nRetrieved {len(retrieved)} documents:")
            for doc, score in retrieved:
                print(f"  [{score:.3f}] {doc['title']}: "
                      f"{doc['content'][:80]}...")
        else:
            print("  No documents retrieved.")

        # Build prompt
        prompt = self.build_prompt(question, retrieved)
        return prompt


# ============================================================
# Exercise 5: Evaluation Helpers
# ============================================================

def evaluate_retrieval(rag: RAGPipeline, test_queries: List[dict]):
    """
    Evaluate retrieval quality on test queries.

    Parameters
    ----------
    rag : RAGPipeline
    test_queries : list of dict
        Each dict has 'question' and 'expected_doc_ids' keys.
    """
    print("\n" + "=" * 50)
    print("Retrieval Evaluation")
    print("=" * 50)

    hits = 0
    total = 0

    for tq in test_queries:
        question = tq["question"]
        expected_ids = set(tq["expected_doc_ids"])

        retrieved = rag.retrieve(question, top_k=3)
        retrieved_ids = {doc["id"] for doc, _ in retrieved}

        # Check if any expected document was retrieved
        found = bool(expected_ids & retrieved_ids)
        if found:
            hits += 1
        total += 1

        status = "HIT" if found else "MISS"
        print(f"\n  [{status}] Q: {question}")
        print(f"    Expected: {expected_ids}")
        print(f"    Retrieved: {retrieved_ids}")

    if total > 0:
        print(f"\n  Recall@3: {hits}/{total} ({100*hits/total:.0f}%)")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    # --- Exercise 1: Document Embedding ---
    print("=" * 60)
    print("Exercise 1: Document Embedding")
    print("=" * 60)

    embedder = DocumentEmbedder()

    # Prepare documents (without chunking first)
    docs = prepare_knowledge_base(KNOWLEDGE_BASE, chunking=False)
    print(f"\nPrepared {len(docs)} documents")

    # Embed all document contents
    texts = [d["content"] for d in docs]
    embeddings = embedder.embed(texts)
    print(f"Embeddings shape: {embeddings.shape}")

    # Show similarity between related documents
    if embeddings is not None:
        # Compute pairwise cosine similarity
        norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        sim_matrix = np.dot(norms, norms.T)
        print("\nSample similarities:")
        for i in range(min(3, len(docs))):
            for j in range(i+1, min(5, len(docs))):
                print(f"  sim('{docs[i]['title']}', '{docs[j]['title']}') "
                      f"= {sim_matrix[i][j]:.3f}")

    # --- Exercise 2: Vector Store ---
    print("\n" + "=" * 60)
    print("Exercise 2: Simple Vector Store")
    print("=" * 60)

    store = SimpleVectorStore()
    store.add(docs, embeddings)

    # Test search
    test_query = "What is the warranty for the robotic arm?"
    query_emb = embedder.embed([test_query])[0]
    results = store.search(query_emb, top_k=3)
    print(f"\nSearch: \"{test_query}\"")
    for doc, score in results:
        print(f"  [{score:.3f}] {doc['title']}")

    # --- Exercise 3: Chunking ---
    print("\n" + "=" * 60)
    print("Exercise 3: Document Chunking")
    print("=" * 60)

    chunked_docs = prepare_knowledge_base(
        KNOWLEDGE_BASE, chunking=True, chunk_size=50, overlap=10
    )
    print(f"Original docs: {len(KNOWLEDGE_BASE)}")
    print(f"After chunking: {len(chunked_docs)}")

    # Show a few chunks
    for chunk in chunked_docs[:3]:
        print(f"  [{chunk['id']}] {chunk['content'][:60]}...")

    # --- Exercise 4: RAG Pipeline ---
    print("\n" + "=" * 60)
    print("Exercise 4: RAG Pipeline")
    print("=" * 60)

    rag = RAGPipeline(embedder, store)

    test_questions = [
        "What is the warranty period for the RoboArm Pro?",
        "How much does the NavBot cost?",
        "How do I set up the RoboArm Pro?",
        "What safety certifications does the RoboArm Pro have?",
        "Can I return a custom-configured robot?",
    ]

    for question in test_questions:
        prompt = rag.query(question, top_k=3)
        print(f"\n{'---' * 14}")
        print("AUGMENTED PROMPT (first 500 chars):")
        print(prompt[:500])
        print("...")

    # --- Exercise 5: Evaluation ---
    print("\n" + "=" * 60)
    print("Exercise 5: Evaluate Retrieval")
    print("=" * 60)

    eval_queries = [
        {
            "question": "What warranty does the RoboArm Pro have?",
            "expected_doc_ids": {"warranty-001"},
        },
        {
            "question": "How much does the RoboArm Lite cost?",
            "expected_doc_ids": {"pricing-001"},
        },
        {
            "question": "What sensors does the NavBot use?",
            "expected_doc_ids": {"spec-002", "prod-003"},
        },
        {
            "question": "How do I get technical support?",
            "expected_doc_ids": {"support-001"},
        },
        {
            "question": "What are the safety requirements?",
            "expected_doc_ids": {"safety-001"},
        },
    ]

    evaluate_retrieval(rag, eval_queries)

    print("\n" + "=" * 60)
    print("Lab 4.5 Complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  - Sentence embeddings map text to dense vectors for semantic search")
    print("  - Vector stores enable fast similarity search over document collections")
    print("  - Chunking improves retrieval by creating focused, topical units")
    print("  - RAG grounds LLM responses in actual documents, reducing hallucination")
    print("  - Retrieval quality is the most important factor in RAG system quality")
    print("\nNext step: Send the augmented prompts to an LLM (OpenAI, Claude, or local)")
    print("and compare answers with and without retrieved context!")
