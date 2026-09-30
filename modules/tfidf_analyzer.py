"""
TF-IDF (Term Frequency - Inverse Document Frequency) Analyzer Module.
Demonstrates classical text mining feature representation using scikit-learn.
"""

from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Reference sentiment & conversational background documents to provide robust IDF weights
DEFAULT_REFERENCE_CORPUS = [
    "I absolutely love this amazing product and fantastic customer service.",
    "This was a terrible, horrible experience with delay and broken delivery.",
    "The meeting is scheduled at 10 AM tomorrow in conference room B.",
    "I feel so happy, delighted, and joyful today with this outstanding achievement.",
    "Very sad and disappointed with the poor quality and rude staff.",
    "What a completely unacceptable and frustrating issue with my order.",
    "Great performance, seamless user experience, and super fast shipping.",
    "Average item, nothing special, neither good nor bad standard quality.",
    "Excited for the upcoming movie and weekend party with friends.",
    "Terrible customer care, no response for three days, highly upset."
]


class TfidfAnalyzer:
    """
    Extracts numerical TF-IDF feature representations and ranks important terms.
    """

    def __init__(self, reference_corpus: List[str] = None):
        self.reference_corpus = reference_corpus or DEFAULT_REFERENCE_CORPUS
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            max_features=100
        )
        # Fit on reference background corpus
        self.vectorizer.fit(self.reference_corpus)

    def extract_top_terms(self, processed_text: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Extracts top-k terms with their TF-IDF scores for the provided processed text.
        """
        if not processed_text or not processed_text.strip():
            return []

        # Combine text into temporary local corpus to get document-specific TF-IDF
        try:
            # We fit a document-level vectorizer if the reference corpus misses unique input words,
            # or transform using fitted vectorizer with fallback
            doc_vectorizer = TfidfVectorizer(
                stop_words='english',
                lowercase=True,
                token_pattern=r'(?u)\b\w+\b',
                ngram_range=(1, 1)
            )
            
            # Use reference + query document for smooth IDF
            combined_corpus = self.reference_corpus + [processed_text]
            tfidf_matrix = doc_vectorizer.fit_transform(combined_corpus)
            
            # The last row is our target document
            query_tfidf = tfidf_matrix[-1].toarray()[0]
            feature_names = np.array(doc_vectorizer.get_feature_names_out())
            
            # Non-zero indices for the query
            nonzero_indices = np.where(query_tfidf > 0)[0]
            if len(nonzero_indices) == 0:
                return []
                
            sorted_indices = nonzero_indices[np.argsort(query_tfidf[nonzero_indices])[::-1]]
            
            top_terms = []
            for idx in sorted_indices[:top_k]:
                top_terms.append({
                    'term': str(feature_names[idx]),
                    'score': round(float(query_tfidf[idx]), 4),
                    'percentage': round(float(query_tfidf[idx]) * 100, 1)
                })
            
            return top_terms
            
        except Exception as e:
            # Simple fallback if vectorization fails
            words = processed_text.split()
            unique_words = list(dict.fromkeys(words))[:top_k]
            return [{'term': w, 'score': 1.0 / len(unique_words), 'percentage': round(100.0 / len(unique_words), 1)} for w in unique_words]
