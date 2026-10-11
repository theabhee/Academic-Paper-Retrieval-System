import os
import pandas as pd
import numpy as np
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    """
    Reranker using a pre-trained Cross-Encoder model (ms-marco-MiniLM-L-6-v2).
    Evaluates query-document pairs jointly to predict high-accuracy relevance scores.
    """
    def __init__(self, df=None, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512):
        self.df = df
        self.model_name = model_name
        self.max_length = max_length
        self._model = None

    @property
    def model(self):
        if self._model is None:
            print(f"Loading Cross-Encoder model '{self.model_name}'...")
            self._model = CrossEncoder(self.model_name, max_length=self.max_length)
        return self._model

    def score_pairs(self, query: str, abstracts: list, batch_size: int = 32) -> np.ndarray:
        """
        Computes relevance scores for (query, abstract) pairs using the Cross-Encoder.
        """
        if not abstracts:
            return np.array([])
        
        pairs = [[query, str(abstract)] for abstract in abstracts]
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        return np.array(scores)

    def rerank(self, query: str, candidates: list, top_k: int = None) -> list:
        """
        Reranks a list of candidate documents.
        Candidates can be:
          - A list of indices: [idx0, idx1, ...]
          - A list of tuples: [(idx0, prev_score), (idx1, prev_score), ...]
          - A list of dicts with 'abstract' or 'index' keys
        
        Returns a list of tuples: [(idx, cross_encoder_score), ...] sorted descending by score.
        """
        if not candidates:
            return []

        # Extract document indices and abstracts
        extracted_indices = []
        abstracts = []

        for item in candidates:
            if isinstance(item, tuple):
                idx = item[0]
            elif isinstance(item, dict):
                idx = item.get("index", item.get("id"))
            else:
                idx = item

            extracted_indices.append(idx)
            if self.df is not None and idx in self.df.index:
                abstracts.append(str(self.df.loc[idx].get("abstract", "")))
            elif isinstance(item, dict) and "abstract" in item:
                abstracts.append(str(item["abstract"]))
            else:
                abstracts.append("")

        scores = self.score_pairs(query, abstracts)

        # Pair indices with cross-encoder scores and sort descending
        ranked = [(int(idx), float(score)) for idx, score in zip(extracted_indices, scores)]
        ranked.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked

    def search(self, query: str, candidate_indices: list = None, top_k: int = 10) -> list:
        """
        Evaluates relevance for documents in the dataset (or a candidate subset)
        and returns results sorted by cross-encoder relevance score.
        """
        if self.df is None or self.df.empty:
            raise ValueError("DataFrame `df` must be provided to search.")

        if candidate_indices is not None:
            candidates = candidate_indices
        else:
            # If candidate set is not provided, evaluate against the dataset
            candidates = list(self.df.index)

        return self.rerank(query, candidates, top_k=top_k)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sample_file = os.path.join(BASE_DIR, "sample_data.csv")

    if not os.path.exists(sample_file):
        sample_file = os.path.join(BASE_DIR, "arxiv_data.csv")

    if os.path.exists(sample_file):
        df_sample = pd.read_csv(sample_file)
        reranker = CrossEncoderReranker(df=df_sample)

        test_query = "neural networks attention mechanism"
        print(f"\n--- Testing Cross-Encoder Reranker ---")
        print(f"Query: {test_query}")

        ranked_results = reranker.search(test_query, top_k=5)
        print("\nTop Results:")
        for rank, (idx, score) in enumerate(ranked_results, 1):
            paper = df_sample.iloc[idx]
            print(f"{rank}. [Score: {score:.4f}] {paper.get('title', 'No Title')}")
            print(f"   Abstract: {paper.get('abstract', '')[:120]}...\n")
    else:
        print("No dataset file found to run demo.")
