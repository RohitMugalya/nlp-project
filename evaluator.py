import time
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.core import Settings

class Evaluator:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.results = []
        
    def calculate_hit_at_k(self, retrieved_doc_ids, relevant_doc_ids):
        return 1 if any(doc_id in relevant_doc_ids for doc_id in retrieved_doc_ids) else 0

    def calculate_recall_at_k(self, retrieved_doc_ids, relevant_doc_ids):
        if not relevant_doc_ids: return 0.0
        hits = sum(1 for doc_id in set(retrieved_doc_ids) if doc_id in relevant_doc_ids)
        return hits / len(set(relevant_doc_ids))

    def calculate_precision_at_k(self, retrieved_doc_ids, relevant_doc_ids, k):
        if k == 0: return 0.0
        hits = sum(1 for doc_id in set(retrieved_doc_ids) if doc_id in relevant_doc_ids)
        return hits / k

    def calculate_mrr(self, retrieved_doc_ids, relevant_doc_ids):
        for i, doc_id in enumerate(retrieved_doc_ids):
            if doc_id in relevant_doc_ids:
                return 1.0 / (i + 1)
        return 0.0

    def calculate_grounding_score(self, response, retrieved_texts):
        if not retrieved_texts or not response: return 0.0
        response_emb = self.embedder.encode([response])
        context_embs = self.embedder.encode(retrieved_texts)
        similarities = cosine_similarity(response_emb, context_embs)[0]
        return float(np.mean(similarities))

    def evaluate_hallucination_rate(self, response, retrieved_texts, llm):
        if not retrieved_texts: return 1.0
        context = "\n".join(retrieved_texts)
        prompt = f"""Given the context below, extract the assertions from the response and check if they are supported by the context.
Return ONLY a JSON formatted string with two keys in the root: "total_claims" (number) and "unsupported_claims" (number). Do not include markdown code block syntax.
Context: {context}
Response: {response}"""
        
        try:
            eval_response = llm.complete(prompt)
            res_str = str(eval_response).strip()
            if res_str.startswith("```json"):
                res_str = res_str[7:]
            if res_str.endswith("```"):
                res_str = res_str[:-3]
            res_str = res_str.strip()
            
            data = json.loads(res_str)
            total = data.get("total_claims", 1)
            unsupported = data.get("unsupported_claims", 0)
            if total == 0: return 0.0
            return unsupported / total
        except Exception as e:
            print("Error in eval hallucination:", e)
            return 0.0

    def evaluate_query(self, query, relevant_docs, rag_system, k_values=[1, 2, 3, 4, 5]):
        query_results = []
        max_k = max(k_values)
        
        start_time = time.time()
        retrieved_nodes_max = rag_system.retrieve(query, k=max_k)
        retrieval_latency = time.time() - start_time
        
        for k in k_values:
            retrieved_nodes = retrieved_nodes_max[:k]
            retrieved_doc_mapping = [node.metadata.get("file_name", "unknown") for node in retrieved_nodes]
            retrieved_scores = [getattr(node, 'score', 0) for node in retrieved_nodes]
            retrieved_texts = [node.get_content() for node in retrieved_nodes]
            
            start_gen = time.time()
            response = rag_system.generate_response(query, retrieved_nodes)
            gen_latency = time.time() - start_gen
            
            # Save response to file
            with open(f"response_K{k}.txt", "w", encoding="utf-8") as f:
                f.write(response)
                
            total_latency = retrieval_latency + gen_latency
                
            hit_at_k = self.calculate_hit_at_k(retrieved_doc_mapping, relevant_docs)
            recall_at_k = self.calculate_recall_at_k(retrieved_doc_mapping, relevant_docs)
            precision_at_k = self.calculate_precision_at_k(retrieved_doc_mapping, relevant_docs, k)
            mrr = self.calculate_mrr(retrieved_doc_mapping, relevant_docs)
            
            grounding_score = self.calculate_grounding_score(response, retrieved_texts)
            hallucination_rate = self.evaluate_hallucination_rate(response, retrieved_texts, Settings.llm)
            
            # Document contribution (based on normalized similarity scores)
            doc_contributions = {}
            if retrieved_scores:
                total_score = sum(retrieved_scores)
                if total_score > 0:
                    for doc, score in zip(retrieved_doc_mapping, retrieved_scores):
                        doc_contributions[doc] = doc_contributions.get(doc, 0) + (score / total_score * 100)
                else:
                    for doc in retrieved_doc_mapping:
                        doc_contributions[doc] = doc_contributions.get(doc, 0) + (100 / len(retrieved_doc_mapping))
            
            result = {
                "K": k,
                "query": query,
                "retrieved_docs": retrieved_doc_mapping,
                "similarity_scores": retrieved_scores,
                "hit_at_k": hit_at_k,
                "recall_at_k": recall_at_k,
                "precision_at_k": precision_at_k,
                "mrr": mrr,
                "grounding_score": grounding_score,
                "hallucination_rate": hallucination_rate,
                "latency": total_latency,
                "doc_contributions": doc_contributions
            }
            query_results.append(result)
            
        self.results.extend(query_results)
        
        with open("evaluation_results.json", "w") as f:
            json.dump(self.results, f, indent=4)
            
        return query_results
