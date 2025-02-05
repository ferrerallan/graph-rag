import openai
import faiss
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
import json
import os
from dotenv import load_dotenv

load_dotenv()

class GraphRAG:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.dimension = 1536
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.graph = nx.Graph()
        self.similarity_threshold = 0.8

    def add_documents(self, docs: List[str]):
        embeddings = []
        for doc in docs:
            response = self.client.embeddings.create(
                input=doc, model="text-embedding-ada-002"
            )
            embedding = response.data[0].embedding
            embeddings.append(embedding)
            doc_id = len(self.documents)
            self.documents.append(doc)
            self.graph.add_node(doc_id, text=doc, embedding=embedding)
        
        self._build_graph(embeddings)
        self.index.add(np.array(embeddings).astype('float32'))

    def _build_graph(self, embeddings: List[List[float]]):
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = self._calculate_similarity(embeddings[i], embeddings[j])
                if similarity > self.similarity_threshold:
                    self.graph.add_edge(i, j, weight=similarity)

    def _calculate_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def _get_subgraph_context(self, doc_indices: List[int], depth: int = 1) -> List[str]:
        context_nodes = set()
        for idx in doc_indices:
            neighbors = set()
            current_nodes = {idx}
            
            for _ in range(depth):
                for node in current_nodes:
                    neighbors.update(self.graph.neighbors(node))
                current_nodes = neighbors - context_nodes
                context_nodes.update(current_nodes)
                neighbors = set()

        return [self.documents[i] for i in context_nodes]

    def query(self, question: str, k: int = 3, graph_depth: int = 1) -> Dict:
        question_embedding = self.client.embeddings.create(
            input=question, model="text-embedding-ada-002"
        ).data[0].embedding

        D, I = self.index.search(np.array([question_embedding]).astype('float32'), k)
        initial_docs = [self.documents[i] for i in I[0]]
        
        # Get additional context from graph
        context = self._get_subgraph_context(I[0], depth=graph_depth)
        
        response = self.generate_response(question, context)
        evaluation = self.evaluate_response(question, response, context)
        final_response = self.refine_response(question, context, response, evaluation)
        
        return {
            "initial_docs": initial_docs,
            "graph_context": context,
            "initial_response": response,
            "evaluation": evaluation,
            "final_response": final_response
        }

    def generate_response(self, question: str, context: List[str]) -> str:
        prompt = f"""
        Context: {' '.join(context)}
        Question: {question}
        Generate a response based only on the provided context.
        """
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant that answers based only on the provided context."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def evaluate_response(self, question: str, response: str, context: List[str]) -> Dict:
        eval_prompt = f"""
        Evaluate this response:
        Question: {question}
        Context: {context}
        Response: {response}
        
        Rate (1-10):
        1. Relevance to question
        2. Use of context
        3. Accuracy
        
        Return JSON format: {{"scores": {{"relevance": X, "context_usage": X, "accuracy": X}}, "suggestions": "improvement notes"}}
        """
        evaluation = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a critical evaluator."}, 
                {"role": "user", "content": eval_prompt}
            ]
        )
        return json.loads(evaluation.choices[0].message.content)

    def refine_response(self, question: str, context: List[str], original_response: str, evaluation: Dict) -> str:
        if sum(evaluation['scores'].values()) / 3 >= 9.5:
            return original_response
            
        refine_prompt = f"""
        Original question: {question}
        Context: {context}
        Original response: {original_response}
        Evaluation: {evaluation}
        
        Provide an improved response addressing the suggestions.
        """
        refined = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Improve the response based on evaluation."},
                {"role": "user", "content": refine_prompt}
            ]
        )
        return refined.choices[0].message.content

if __name__ == "__main__":
    api = os.getenv("OPENAI_KEY")
    
    rag = GraphRAG(api)
    
    docs = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Python was created by Guido van Rossum and was first released in 1991.",
        "Python is widely used in data science and machine learning applications.",
        "Machine learning algorithms require significant computational resources.",
        "Data science involves analyzing and interpreting complex data sets.",
        "OpenAI develops advanced language models using Python.",
        "Python's popularity in AI is due to its extensive libraries."
    ]
    rag.add_documents(docs)
    
    question = "How is Python used in data science?"
    result = rag.query(question)
    
    print("\nInitial docs:", result["initial_docs"])
    print("\nGraph context:", result["graph_context"])
    print("\nInitial response:", result["initial_response"])
    print("\nEvaluation:", result["evaluation"])
    print("\nFinal response:", result["final_response"])