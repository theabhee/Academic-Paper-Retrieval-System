import os
import time
import math
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from models import TfidfRetrieval, BM25Retrieval, SemanticRetrieval
from reranker import CrossEncoderReranker

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GROUND_TRUTH = {
    "neural networks attention": [1, 2, 6],
    "image recognition deep learning": [5],
    "generative models adversarial": [4],
    "word embeddings semantic relationships": [3]
}

def calculate_metrics(retrieved_indices, relevant_indices):
    """
    Calculates Precision, Recall, and F1-score for retrieved documents.
    Preserves backward compatibility with existing callers.
    """
    retrieved_set = set(retrieved_indices)
    relevant_set = set(relevant_indices)
    
    true_positives = len(retrieved_set.intersection(relevant_set))
    
    precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
    recall = true_positives / len(relevant_set) if relevant_set else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def calculate_ndcg_at_k(retrieved_indices, relevant_indices, k=3):
    """
    Calculates Normalized Discounted Cumulative Gain at rank k (nDCG@k).
    Uses binary relevance (1 if document is in relevant_indices, else 0).
    """
    rel_set = set(relevant_indices)
    retrieved_k = retrieved_indices[:k]
    
    dcg = 0.0
    for i, idx in enumerate(retrieved_k):
        if idx in rel_set:
            # i=0 corresponds to rank 1 -> log2(1 + 1) = log2(2) = 1.0
            dcg += 1.0 / math.log2(i + 2)
            
    num_relevant = min(k, len(relevant_indices))
    if num_relevant == 0:
        return 0.0
        
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate():
    try:
        data_path = os.path.join(BASE_DIR, "sample_data.csv")
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Error: sample_data.csv not found.")
        return

    print("=" * 80)
    print("ACADEMIC PAPER RETRIEVAL: EVALUATING ORIGINAL VS. ENHANCED SYSTEM")
    print("=" * 80)
    print("Initializing base retrieval models and Cross-Encoder reranker...")

    # Initialize original base models
    tfidf_model = TfidfRetrieval(df)
    bm25_model = BM25Retrieval(df)
    semantic_model = SemanticRetrieval(df)
    
    # Initialize Cross-Encoder reranker
    reranker = CrossEncoderReranker(df)
    _ = reranker.model  # Pre-warm model weights

    # Define system retrieval runners
    def run_tfidf(query):
        return tfidf_model.search(query)

    def run_bm25(query):
        return bm25_model.search(query)

    def run_semantic(query):
        return semantic_model.search(query)

    def run_tfidf_reranked(query):
        candidates = tfidf_model.search(query)[:20]
        return reranker.rerank(query, candidates, top_k=3)

    def run_bm25_reranked(query):
        candidates = bm25_model.search(query)[:20]
        return reranker.rerank(query, candidates, top_k=3)

    def run_semantic_reranked(query):
        candidates = semantic_model.search(query)[:20]
        return reranker.rerank(query, candidates, top_k=3)

    def run_cross_encoder_direct(query):
        return reranker.search(query, top_k=3)

    systems = {
        "TF-IDF": (run_tfidf, "Original"),
        "BM25": (run_bm25, "Original"),
        "Semantic": (run_semantic, "Original"),
        "TF-IDF + Reranker": (run_tfidf_reranked, "Enhanced"),
        "BM25 + Reranker": (run_bm25_reranked, "Enhanced"),
        "Semantic + Reranker": (run_semantic_reranked, "Enhanced"),
        "Cross-Encoder Direct": (run_cross_encoder_direct, "Enhanced")
    }

    all_results = []
    summary_metrics = []

    for system_name, (runner, system_type) in systems.items():
        p_list, r_list, ndcg_list, latency_list = [], [], [], []

        for query, relevant_ids in GROUND_TRUTH.items():
            relevant_indices = [i - 1 for i in relevant_ids]

            # Measure latency in milliseconds
            start_time = time.perf_counter()
            results = runner(query)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            retrieved_indices = [idx for idx, _ in results[:3]]

            p, r, _ = calculate_metrics(retrieved_indices, relevant_indices)
            ndcg = calculate_ndcg_at_k(retrieved_indices, relevant_indices, k=3)

            p_list.append(p)
            r_list.append(r)
            ndcg_list.append(ndcg)
            latency_list.append(latency_ms)

            all_results.append({
                "System": system_name,
                "Type": system_type,
                "Query": query,
                "Precision@3": p,
                "Recall@3": r,
                "nDCG@3": ndcg,
                "Latency_ms": latency_ms
            })

        avg_p = np.mean(p_list)
        avg_r = np.mean(r_list)
        avg_ndcg = np.mean(ndcg_list)
        avg_lat = np.mean(latency_list)

        summary_metrics.append({
            "System": system_name,
            "Type": system_type,
            "Precision@3": avg_p,
            "Recall@3": avg_r,
            "nDCG@3": avg_ndcg,
            "Avg_Latency_ms": avg_lat
        })

    # Display comparison table
    print("\n" + "=" * 90)
    print(f"| {'System Name':<22} | {'Type':<10} | {'Precision@3':<12} | {'Recall@3':<10} | {'nDCG@3':<10} | {'Latency (ms)':<12} |")
    print("=" * 90)
    for row in summary_metrics:
        print(f"| {row['System']:<22} | {row['Type']:<10} | {row['Precision@3']:<12.4f} | {row['Recall@3']:<10.4f} | {row['nDCG@3']:<10.4f} | {row['Avg_Latency_ms']:<12.2f} |")
    print("=" * 90)

    # Save detailed evaluation dataframe
    df_results = pd.DataFrame(all_results)
    csv_path = os.path.join(BASE_DIR, "evaluation_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n[Saved] Detailed per-query results saved to: '{csv_path}'")

    summary_df = pd.DataFrame(summary_metrics)
    summary_csv_path = os.path.join(BASE_DIR, "evaluation_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[Saved] Summary comparison table saved to: '{summary_csv_path}'")

    # Generate Visualization: Metrics comparison box plot & bar chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Metric scores comparison boxplot
    melted_df = pd.melt(
        df_results,
        id_vars=['System', 'Type'],
        value_vars=['Precision@3', 'Recall@3', 'nDCG@3'],
        var_name='Metric',
        value_name='Score'
    )

    sns.boxplot(
        data=melted_df,
        x='Metric',
        y='Score',
        hue='System',
        palette='tab10',
        ax=axes[0]
    )
    axes[0].set_title('Retrieval Quality: Precision@3, Recall@3 & nDCG@3', fontsize=12, fontweight='bold')
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc='lower left', fontsize=8)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # Response latency comparison bar chart
    sns.barplot(
        data=summary_df,
        x='System',
        y='Avg_Latency_ms',
        hue='Type',
        palette={'Original': '#4C72B0', 'Enhanced': '#C44E52'},
        ax=axes[1]
    )
    axes[1].set_title('Response Latency Cost (Average ms per Query)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Latency (ms)')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=35, ha='right', fontsize=9)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    chart_path = os.path.join(BASE_DIR, "results_boxplot.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[Saved] Comparison visualizations saved to: '{chart_path}'\n")

    return summary_metrics

if __name__ == "__main__":
    evaluate()
