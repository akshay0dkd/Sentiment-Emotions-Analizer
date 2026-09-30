"""
Text Preprocessing Module.
Handles Unicode normalization, tokenization, lemmatization, stop-word removal,
and dual-stream text preparation (Model Stream vs Text Mining Stream).
"""

import re
import unicodedata
from typing import Dict, List, Any, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import os
import warnings

# Suppress NLTK world-writable warnings on cloud containers
warnings.filterwarnings('ignore', category=UserWarning, module='nltk')

# Ensure required NLTK resources are available in a safe private user directory
def _download_nltk_resources():
    try:
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        if nltk_data_dir not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data_dir)
    except Exception:
        nltk_data_dir = None

    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            if nltk_data_dir:
                nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
            else:
                nltk.download(resource, quiet=True)
        except Exception:
            pass

_download_nltk_resources()


class TextPreprocessor:
    """
    Robust Dual-Stream Text Preprocessor for NLP and Text Mining.
    
    Maintains:
    1. Model Stream: Cleaned whitespace & unicode normalized, but retains emojis, casing,
       and punctuation essential for Transformer contextual representations.
    2. Linguistic/Mining Stream: Lowercased, stripped punctuation, filtered stopwords,
       and lemmatized for classical bag-of-words and TF-IDF feature analysis.
    """

    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            # Fallback essential stopwords list if NLTK corpus download is constrained
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
                "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
                'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
                'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
                'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
                'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
                'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
                'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
                'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
                'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
                'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
                "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
                'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
            }
        
        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception:
            self.lemmatizer = None

    def normalize_unicode(self, text: str) -> str:
        """Applies NFKC Unicode normalization."""
        return unicodedata.normalize('NFKC', text)

    def extract_metadata(self, text: str) -> Dict[str, List[str]]:
        """Extracts URLs, mentions, and hashtags without deleting them."""
        urls = re.findall(r'https?://\S+|www\.\S+', text)
        mentions = re.findall(r'@\w+', text)
        hashtags = re.findall(r'#\w+', text)
        return {
            'urls': urls,
            'mentions': mentions,
            'hashtags': hashtags
        }

    def prepare_model_input(self, text: str) -> str:
        """
        Prepares text for Transformer models.
        Normalizes Unicode & excess whitespace, but preserves emojis, casing, and punctuation.
        """
        if not text:
            return ""
        norm_text = self.normalize_unicode(text)
        # Collapse multiple spaces and trim
        norm_text = re.sub(r'[ \t\r\f\v]+', ' ', norm_text).strip()
        return norm_text

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes text using NLTK word_tokenize with regex fallback."""
        try:
            return word_tokenize(text)
        except Exception:
            # Fallback regex word tokenization preserving words and punctuation
            return re.findall(r'\w+|[^\w\s]', text, re.UNICODE)

    def lemmatize_token(self, token: str) -> str:
        """Lemmatizes a single token."""
        if not self.lemmatizer:
            return token.lower()
        try:
            # Default to verb then noun lemmatization
            lemma = self.lemmatizer.lemmatize(token.lower(), pos='v')
            if lemma == token.lower():
                lemma = self.lemmatizer.lemmatize(token.lower(), pos='n')
            return lemma
        except Exception:
            return token.lower()

    def process_for_text_mining(self, text: str) -> Dict[str, Any]:
        """
        Executes full NLP pipeline for Text Mining & Feature Extraction.
        """
        raw_text = self.normalize_unicode(text or "")
        metadata = self.extract_metadata(raw_text)
        
        # Tokenize original text
        raw_tokens = self.tokenize(raw_text)
        
        # Lowercase tokens for linguistic analysis
        clean_tokens = [t.lower() for t in raw_tokens if re.match(r'^[a-zA-Z0-9_-]+$', t)]
        
        # Stop-words identification
        stop_words_found = [t for t in clean_tokens if t in self.stop_words]
        non_stop_tokens = [t for t in clean_tokens if t not in self.stop_words]
        
        # Lemmatization
        lemmatized_tokens = [self.lemmatize_token(t) for t in non_stop_tokens]
        
        # Reconstructed cleaned text for TF-IDF
        processed_text = " ".join(lemmatized_tokens)
        
        return {
            'original_text': text,
            'model_input_text': self.prepare_model_input(text),
            'processed_text': processed_text,
            'raw_tokens': raw_tokens,
            'clean_tokens': clean_tokens,
            'stop_words_found': stop_words_found,
            'lemmas': lemmatized_tokens,
            'metadata': metadata
        }
