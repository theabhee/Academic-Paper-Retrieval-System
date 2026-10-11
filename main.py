import pandas as pd
import argparse
from models import BooleanRetrieval, TfidfRetrieval, BM25Retrieval, SemanticRetrieval

def display_results(df, results, top_k=5):
    if not results:
        print("No matches found.")
        return
        
    print(f"\nTop {min(top_k, len(results))} Results:\n")
    for i, (idx, score) in enumerate(results[:top_k]):
        print(f"{i+1}. Score: {score:.4f}")
        print(f"Title: {df.iloc[idx]['title']}")
        print(f"Abstract: {df.iloc[idx]['abstract'][:200]}...\n")

def main():
    parser = argparse.ArgumentParser(description="Semantic Academic Paper Retrieval System")
    parser.add_argument("--query", type=str, default="machine learning", help="Search query (default: 'machine learning')")
    parser.add_argument("--method", type=str, choices=['boolean', 'tfidf', 'bm25', 'semantic'], default='tfidf', help="Retrieval method (default: tfidf)")
    parser.add_argument("--data", type=str, default="arxiv_data.csv", help="Path to CSV data file (default: arxiv_data.csv)")
    
    args = parser.parse_args()

    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = args.data
    if not os.path.exists(data_path):
        alt_path = os.path.join(BASE_DIR, args.data)
        if os.path.exists(alt_path):
            data_path = alt_path
        else:
            sample_path = os.path.join(BASE_DIR, "sample_data.csv")
            if os.path.exists(sample_path):
                data_path = sample_path

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file '{args.data}' not found.")
        return

    print(f"Initializing {args.method} retrieval system...")
    if args.method == 'boolean':
        model = BooleanRetrieval(df)
    elif args.method == 'tfidf':
        model = TfidfRetrieval(df)
    elif args.method == 'bm25':
        model = BM25Retrieval(df)
    elif args.method == 'semantic':
        model = SemanticRetrieval(df)
        
    print(f"\nSearching for: '{args.query}'")
    results = model.search(args.query)
    
    display_results(df, results)

if __name__ == "__main__":
    main()


#python main.py --query "superconducting qubits coherence" --method boolean
#python main.py --query "Artificial Intelligence" --method bm25
#python main.py --query "machine learning" --method tfidf3