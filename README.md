# Semantic Academic Paper Retrieval System

This project is a complete information retrieval system for academic research papers, implementing various techniques to search and rank documents based on a user's query.

## Overview

The system processes a dataset of academic papers (titles and abstracts), applying standard NLP preprocessing (tokenization, stop-word removal, lemmatization). It supports four different retrieval strategies:

1. **Boolean Retrieval**: Basic exact-match search (AND logic).
2. **TF-IDF**: Term Frequency-Inverse Document Frequency using Cosine Similarity for vector-space ranking.
3. **BM25**: A highly effective probabilistic retrieval model.
4. **Semantic Search**: Vector-based semantic retrieval using `sentence-transformers`, allowing the system to match concepts and meaning even if exact keywords are absent.

## Installation

1. Make sure you have Python 3.8+ installed.
2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

*(Note: The system will automatically download necessary NLTK corpora, such as WordNet and stopwords, the first time you run it.)*

## Usage

### Searching

Use `main.py` to run queries against the dataset (`sample_data.csv`).

**Basic syntax:**
```bash
python main.py --query "YOUR_SEARCH_QUERY" --method [boolean|tfidf|bm25|semantic]
```

**Examples:**

Using BM25:
```bash
python main.py --query "deep learning models" --method bm25
```

Using Semantic Search:
```bash
python main.py --query "generative image creation" --method semantic
```

### Evaluation

Run `evaluation.py` to see the performance of TF-IDF, BM25, and Semantic models on a set of predefined ground-truth queries, measured using standard IR metrics (Precision, Recall, F1-Score).

```bash
python evaluation.py
```

## Project Structure
- `preprocessing.py`: Handles text tokenization, stopwords, and lemmatization via NLTK.
- `models.py`: Contains the logic for the different retrieval approaches.
- `main.py`: Command-line interface for the search engine.
- `evaluation.py`: Script to compute precision, recall, and F1-score.
- `sample_data.csv`: A sample dataset containing titles and abstracts of famous ML papers.
