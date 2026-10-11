import json
import pandas as pd
import os

input_file = 'arxiv-metadata-oai-snapshot.json'
output_file = 'arxiv_data.csv'
num_papers = 10000

print(f"Extracting {num_papers} papers from {input_file}...")

papers = []
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_papers:
                break
                
            paper = json.loads(line)
            papers.append({
                'id': paper.get('id', str(i)),
                'title': paper.get('title', '').replace('\n', ' ').strip(),
                'abstract': paper.get('abstract', '').replace('\n', ' ').strip()
            })
            
    df = pd.DataFrame(papers)
    df.to_csv(output_file, index=False)
    print(f"Successfully saved {len(df)} papers to {output_file}!")
    
except Exception as e:
    print(f"An error occurred: {e}")
