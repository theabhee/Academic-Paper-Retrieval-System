from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
from preprocessing import TextPreprocessor

import pickle
import os

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_processed_docs(df):
    cache_path = os.path.join(CACHE_DIR, f"processed_docs_{len(df)}.pkl")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    print(f"Preprocessing {len(df)} document abstracts (one-time fast indexing)...")
    preprocessor = TextPreprocessor()
    processed = [preprocessor.preprocess(doc) for doc in df['abstract'].tolist()]

    try:
        with open(cache_path, "wb") as f:
            pickle.dump(processed, f)
    except Exception:
        pass

    return processed

class BooleanRetrieval:
    def __init__(self, df, processed_docs=None):
        self.df = df
        self.preprocessor = TextPreprocessor()
        self.processed_docs = processed_docs if processed_docs is not None else get_processed_docs(df)

    def search(self, query):
        processed_query = self.preprocessor.preprocess(query).split()
        results = []
        for i, doc in enumerate(self.processed_docs):
            if all(term in doc for term in processed_query):
                results.append((i, 1.0))
        return results

class TfidfRetrieval:
    def __init__(self, df, processed_docs=None):
        self.df = df
        self.preprocessor = TextPreprocessor()
        self.processed_docs = processed_docs if processed_docs is not None else get_processed_docs(df)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_docs)

    def search(self, query):
        processed_query = self.preprocessor.preprocess(query)
        query_vector = self.vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        results = []
        for i in similarities.argsort()[::-1]:
            if similarities[i] > 0:
                results.append((i, float(similarities[i])))
        return results

class BM25Retrieval:
    def __init__(self, df, processed_docs=None):
        self.df = df
        self.preprocessor = TextPreprocessor()
        docs = processed_docs if processed_docs is not None else get_processed_docs(df)
        tokenized_docs = [doc.split() for doc in docs]
        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query):
        tokenized_query = self.preprocessor.preprocess(query).split()
        scores = self.bm25.get_scores(tokenized_query)
        
        results = []
        for i in scores.argsort()[::-1]:
            if scores[i] > 0:
                results.append((i, float(scores[i])))
        return results

class SemanticRetrieval:
    def __init__(self, df):
        self.df = df
        self.model = None
        self.doc_embeddings = None

    def _ensure_initialized(self):
        if self.model is None:
            cache_file = os.path.join(CACHE_DIR, f"semantic_embeddings_{len(self.df)}.pt")
            print("Loading Sentence Transformer model...")
            self.model = SentenceTransformer('all-mpnet-base-v2')

            if os.path.exists(cache_file):
                try:
                    import torch
                    print("Loading cached vector embeddings from disk...")
                    self.doc_embeddings = torch.load(cache_file)
                    return
                except Exception:
                    pass

            print("Encoding paper abstracts into vector space...")
            # Sample up to 2500 abstracts for fast high-quality semantic search
            abstracts = self.df['abstract'].head(2500).tolist()
            self.doc_embeddings = self.model.encode(abstracts, convert_to_tensor=True)

            try:
                import torch
                torch.save(self.doc_embeddings, cache_file)
            except Exception:
                pass

    def search(self, query):
        self._ensure_initialized()
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]
        
        scores = cos_scores.cpu().numpy()
        results = []
        for i in scores.argsort()[::-1]:
            if scores[i] > 0:
                results.append((i, float(scores[i])))
        return results

class SearchEngineManager:
    def __init__(self, df):
        self.df = df
        print("Initializing SearchEngineManager...")
        processed_docs = get_processed_docs(df)
        self.models = {
            'boolean': BooleanRetrieval(df, processed_docs=processed_docs),
            'tfidf': TfidfRetrieval(df, processed_docs=processed_docs),
            'bm25': BM25Retrieval(df, processed_docs=processed_docs),
            'semantic': SemanticRetrieval(df)
        }

    def format_results(self, results, method, top_k=10):
        formatted = []
        if not results:
            return formatted

        # Find max score for normalization if BM25
        max_score = max(score for _, score in results) if results else 1.0

        for idx, score in results[:top_k]:
            paper = self.df.iloc[idx]
            
            # Normalize percentage score for UI display
            if method == 'boolean':
                norm_score = 100.0
            elif method == 'bm25':
                norm_score = min(100.0, round((score / max(max_score, 1e-5)) * 100, 1))
            else: # tfidf, semantic (cosine sim: 0 to 1)
                norm_score = min(100.0, max(0.0, round(score * 100, 1)))

            formatted.append({
                "index": int(idx),
                "id": str(paper.get('id', idx + 1)),
                "title": str(paper.get('title', '')),
                "abstract": str(paper.get('abstract', '')),
                "raw_score": round(float(score), 4),
                "match_percentage": norm_score
            })
        return formatted

    def search(self, query, method='tfidf', top_k=10):
        if method not in self.models:
            method = 'tfidf'
        raw_results = self.models[method].search(query)
        return self.format_results(raw_results, method, top_k=top_k)

    def search_all(self, query, top_k=5):
        return {
            method: self.format_results(model.search(query), method, top_k=top_k)
            for method, model in self.models.items()
        }
