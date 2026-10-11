import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from functools import lru_cache

# Download necessary NLTK data (quietly)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    @lru_cache(maxsize=20000)
    def _lemmatize_word(self, word):
        return self.lemmatizer.lemmatize(word)

    def preprocess(self, text):
        if not isinstance(text, str):
            return ""
            
        # Fast C-level regex extraction of lowercased words
        tokens = re.findall(r'\b[a-z]{2,}\b', text.lower())
        
        # Remove stop words and lemmatize using cached lookup
        processed_tokens = [
            self._lemmatize_word(w) 
            for w in tokens 
            if w not in self.stop_words
        ]
        
        return " ".join(processed_tokens)
