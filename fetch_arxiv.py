import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd

print("Fetching papers from ArXiv API...")
# We fetch 200 recent papers from the Computer Science / AI category
url = 'http://export.arxiv.org/api/query?search_query=cat:cs.AI&start=0&max_results=200'

try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    data = urllib.request.urlopen(req).read()
    root = ET.fromstring(data)

    ns = {'atom': 'http://www.w3.org/2005/Atom'}

    papers = []
    for idx, entry in enumerate(root.findall('atom:entry', ns)):
        title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
        summary = entry.find('atom:summary', ns).text.replace('\n', ' ').strip()
        
        papers.append({
            'id': idx + 1,
            'title': title,
            'abstract': summary
        })

    df = pd.DataFrame(papers)
    df.to_csv('arxiv_data.csv', index=False)
    print(f"Successfully saved {len(papers)} ArXiv papers to arxiv_data.csv")
except Exception as e:
    print(f"Failed to fetch data: {e}")
