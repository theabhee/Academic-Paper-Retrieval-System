import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models import TfidfRetrieval, BM25Retrieval, SemanticRetrieval
import warnings
warnings.filterwarnings('ignore') 


GROUND_TRUTH = {
    "neural networks attention": [1, 2, 6],
    "image recognition deep learning": [5],
    "generative models adversarial": [4],
    "word embeddings semantic relationships": [3]
}

def calculate_metrics(retrieved_indices, relevant_indices):
    retrieved_set = set(retrieved_indices)
    relevant_set = set(relevant_indices)
    
    true_positives = len(retrieved_set.intersection(relevant_set))
    
    precision = true_positives / len(retrieved_set) if retrieved_set else 0
    recall = true_positives / len(relevant_set) if relevant_set else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def evaluate():
    try:
        data_path = os.path.join(BASE_DIR, "sample_data.csv")
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Error: sample_data.csv not found.")
        return
        
    print("Initializing models for evaluation...")
    models = {
        "TF-IDF": TfidfRetrieval(df),
        "BM25": BM25Retrieval(df),
        "Semantic": SemanticRetrieval(df)
    }
    
    print("\n+" + "-"*16 + "+" + "-"*16 + "+" + "-"*16 + "+" + "-"*16 + "+")
    print(f"| {'Model':<14} | {'Precision@3':<14} | {'Recall@3':<14} | {'F1-Score':<14} |")
    print("+" + "-"*16 + "+" + "-"*16 + "+" + "-"*16 + "+" + "-"*16 + "+")
    
    all_metrics = []

    for model_name, model in models.items():
        total_p, total_r, total_f1 = 0, 0, 0
        
        for query, relevant_ids in GROUND_TRUTH.items():
            relevant_indices = [i - 1 for i in relevant_ids]
            results = model.search(query)
            
          
            retrieved_indices = [idx for idx, score in results[:3]]
            
            p, r, f1 = calculate_metrics(retrieved_indices, relevant_indices)
            total_p += p; total_r += r; total_f1 += f1
            
            all_metrics.append({'Model': model_name, 'Metric': 'Precision@3', 'Score': p})
            all_metrics.append({'Model': model_name, 'Metric': 'Recall@3', 'Score': r})
            all_metrics.append({'Model': model_name, 'Metric': 'F1-Score', 'Score': f1})
            
        n = len(GROUND_TRUTH)
        avg_p = total_p / n
        avg_r = total_r / n
        avg_f1 = total_f1 / n
        
        print(f"| {model_name:<14} | {avg_p:<14.4f} | {avg_r:<14.4f} | {avg_f1:<14.4f} |")
        
    print("+" + "-"*16 + "+" + "-"*16 + "+" + "-"*16 + "+" + "-"*16 + "+")

    metrics_df = pd.DataFrame(all_metrics)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=metrics_df, x='Metric', y='Score', hue='Model')
    plt.title('Distribution of Evaluation Metrics Across Queries (Box Plot)')
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig('results_boxplot.png')
    print("\nBox plot successfully saved as 'results_boxplot.png'!")

if __name__ == "__main__":
    evaluate()
