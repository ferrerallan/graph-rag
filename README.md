# GraphRAG

A Graph-based Retrieval-Augmented Generation (RAG) system that combines vector similarity search with graph-based document relationships for enhanced context retrieval.

## Features

- Vector embeddings using OpenAI's text-embedding-ada-002
- FAISS for efficient similarity search
- NetworkX for graph-based document relationships
- Contextual retrieval using graph traversal
- Response generation and refinement using GPT-4
- Automated response evaluation and improvement

## Installation

```bash
git clone https://github.com/ferrerallan/graph-rag
cd graph-rag
pip install -r requirements.txt
```

Copy the environment file and add your OpenAI API key:
```bash
cp .env.example .env
```

Edit `.env` and replace `xxxx` with your OpenAI API key.

## Required Dependencies

- openai
- faiss-cpu
- numpy
- networkx
- python-dotenv

## Usage

```python
from graph_rag import GraphRAG
import os

# Initialize GraphRAG
rag = GraphRAG(os.getenv("OPENAI_KEY"))

# Add documents
docs = [
    "Python is a high-level programming language known for its simplicity.",
    "Python is widely used in data science applications.",
    # Add more documents...
]
rag.add_documents(docs)

# Query the system
result = rag.query("How is Python used in data science?")

# Access results
print(result["initial_docs"])      # Initially retrieved documents
print(result["graph_context"])     # Additional context from graph
print(result["initial_response"])  # Generated response
print(result["evaluation"])        # Response evaluation
print(result["final_response"])    # Refined response
```

## How It Works

1. **Document Processing**:
   - Converts documents into embeddings using OpenAI
   - Stores embeddings in FAISS index
   - Builds a graph of document relationships

2. **Query Processing**:
   - Finds similar documents using vector search
   - Expands context using graph relationships
   - Generates, evaluates, and refines responses using GPT-4

3. **Response Refinement**:
   - Evaluates response quality
   - Automatically improves responses below quality threshold
   - Returns comprehensive result including evaluation metrics

## Configuration

- `similarity_threshold`: Minimum similarity for document relationship (default: 0.8)
- `k`: Number of initial documents to retrieve (default: 3)
- `graph_depth`: Depth of graph traversal for context (default: 1)

## License

MIT License