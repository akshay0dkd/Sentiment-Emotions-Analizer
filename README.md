# 🧠 Intelligent Sentiment & Emotion Analysis Using Text Mining, NLP, and Transformer Models

An end-to-end, production-grade, and academically rigorous application combining classical Text Mining, NLP Feature Extraction, TF-IDF Vectorization, Unicode Emoji Analysis, and Pretrained Hugging Face Transformers (`Twitter-RoBERTa` & `DistilRoBERTa`) in an interactive **Streamlit** dashboard.

---

## 📌 1. Project Overview & Motivation

In modern digital communication, user-generated texts (customer reviews, tweets, customer support chats, forum comments) contain nuanced sentiment, multi-class emotions, slang, hashtags, and emojis. Traditional rule-based or shallow machine learning approaches often fail to capture contextual semantics and non-verbal emoji cues.

This project delivers a complete dual-stream analytical framework:
1. **Classical Text Mining Stream**: Unicode normalization, tokenization, stop-word removal, WordNet lemmatization, statistical surface metrics, and **TF-IDF vectorization**.
2. **Deep Learning Transformer Stream**: Dense contextual self-attention representations using **Twitter-RoBERTa** (3-class Sentiment) and **DistilRoBERTa** (7-class Ekman Emotion taxonomy), preserving emojis, casing, and sentence structure.
3. **Emoji Semantics Engine**: Independent extraction and semantic mapping of Unicode emojis to emotional intent.

---

## 🏗️ 2. System Architecture

```text
                                User Text Input
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │    Text Preprocessing &       │
                      │    Unicode Normalization      │
                      └───────────────┬───────────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
   ┌───────────────────────────┐             ┌───────────────────────────┐
   │    Classical Text Mining  │             │   Deep Learning Stream    │
   │           Stream          │             │   (Emojis Preserved)      │
   ├───────────────────────────┤             ├───────────────────────────┤
   │ • Tokenization & Stopwords│             │ • HuggingFace Tokenizers  │
   │ • WordNet Lemmatization   │             │ • Twitter-RoBERTa (Senti) │
   │ • Surface Feature Counts  │             │ • DistilRoBERTa (Emotion) │
   │ • TF-IDF Vectorizer       │             │ • Softmax Probabilities   │
   └─────────────┬─────────────┘             └─────────────┬─────────────┘
                 │                                         │
                 │          ┌───────────────────┐          │
                 └─────────►│   Emoji Analysis  │◄─────────┘
                            │   & Polarity Map  │
                            └─────────┬─────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   Result Engine & Alerts  │
                        │ (Mixed Signal Heuristics) │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │    Streamlit Dashboard    │
                        │  (Interactive Plotly Viz) │
                        └───────────────────────────┘
```

---

## 🤖 3. Model Registry & Specifications

| Dimension | Sentiment Model | Emotion Model |
| :--- | :--- | :--- |
| **Model Name** | Twitter-RoBERTa Sentiment Latest | DistilRoBERTa Emotion Classifier |
| **Hugging Face ID** | `cardiffnlp/twitter-roberta-base-sentiment-latest` | `j-hartmann/emotion-english-distilroberta-base` |
| **Base Architecture**| RoBERTa Base (125M parameters) | DistilRoBERTa (82M parameters) |
| **Pretraining Corpus**| ~124M tweets (Jan 2018 – Dec 2021) | 6 Diverse Emotion Datasets |
| **Output Classes** | `Positive`, `Neutral`, `Negative` | `Joy`, `Sadness`, `Anger`, `Fear`, `Surprise`, `Disgust`, `Neutral` |
| **Inference Hardware**| Auto-detected CUDA GPU or CPU | Auto-detected CUDA GPU or CPU |

---

## 🛠️ 4. Tech Stack

- **Application & UI:** Streamlit
- **Deep Learning Framework:** PyTorch & Hugging Face `transformers`
- **Classical NLP & Mining:** NLTK (`punkt`, `stopwords`, `wordnet`), `scikit-learn` (`TfidfVectorizer`)
- **Emoji Semantics:** `emoji` library
- **Visualizations:** Plotly Interactive Graphs & Metrics Cards
- **Testing:** `pytest`

---

## 📁 5. Project Directory Structure

```text
SEntiment Analysis/
│
├── app.py                      # Streamlit application UI & reactive dashboard
│
├── modules/
│   ├── __init__.py
│   ├── preprocessing.py        # Dual-stream NLP, Unicode normalization, Lemmatizer
│   ├── feature_extraction.py   # Surface metrics, lexical cues, contrastive checks
│   ├── tfidf_analyzer.py       # TF-IDF calculation & top keyword ranking
│   ├── emoji_analyzer.py       # Unicode emoji detection, frequency & polarity mapping
│   ├── sentiment_analyzer.py   # Transformer sentiment inference & mixed signal detector
│   ├── emotion_analyzer.py     # Transformer multi-class emotion classification
│   └── visualization.py        # Interactive Plotly charts (bars, donuts, rankings)
│
├── models/
│   ├── __init__.py
│   └── model_loader.py         # Cached model loading (@st.cache_resource) & hardware detection
│
├── data/
│   └── sample_data/
│       ├── test_cases.json     # Academic benchmark test scenarios
│       └── sample_reviews.csv  # Labeled dataset for demonstration and evaluation
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py   # Tests for tokenization, lemmatization, and metadata
│   ├── test_features.py        # Tests for surface text statistics and lexical markers
│   ├── test_emoji.py           # Tests for emoji detection and frequency counts
│   └── test_analysis.py        # Tests for TF-IDF ranking and model pipelines
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview & documentation
├── ACADEMIC_REPORT.md          # Complete 9-Chapter thesis / viva report
└── .gitignore
```

---

## 🚀 6. Installation & Execution

### Step 1: Clone or Navigate to Directory
```bash
cd "c:/Users/sanja/AkshayProjects/SEntiment Analysis"
```

### Step 2: Install Dependencies
```bash
py -3.13 -m pip install -r requirements.txt
```

### Step 3: Run the Streamlit Application
```bash
py -3.13 -m streamlit run app.py
```
The application will launch locally at `http://localhost:8501`.

---

## 🧪 7. Running Unit Tests

Run the complete automated test suite:
```bash
py -3.13 -m pytest tests/ -v
```

---

## 📊 8. Example Inputs & Analysis Results

| Category | Input Sentence | Model Sentiment | Dominant Emotion | Key Features & Emojis |
| :--- | :--- | :--- | :--- | :--- |
| **Positive Review** | `"I absolutely love this product! It exceeded all expectations 😍❤️🔥"` | `Positive (96.4%)` | `Joy (92.1%)` | 3 Emojis (`😍`, `❤️`, `🔥`), TF-IDF: *exceeded, expectations, product* |
| **Service Outrage** | `"This service is horrible, completely unacceptable, and rude! 😡🤬"` | `Negative (98.2%)` | `Anger (94.6%)` | 2 Emojis (`😡`, `🤬`), Exclamations: 1, Lexical Negatives |
| **Sarcastic Delay** | `"Great, another two hour flight delay... wonderful airline service 🙃"` | `Negative (84.1%)` | `Disgust / Annoyance` | Irony emoji (`🙃`), Ellipsis detected, Sarcasm challenge |
| **Mixed / Contrastive** | `"The camera is fantastic, but the battery life is terrible."` | `Mixed / Neutral` | `Neutral / Disgust` | Conjunction: `but`, Positive: `fantastic`, Negative: `terrible` |
| **Factual / Schedule** | `"The project review meeting is scheduled for tomorrow at 10 AM."` | `Neutral (92.5%)` | `Neutral (89.3%)` | Zero emotional tokens, zero emojis |

---

## 💡 9. Academic Viva Preparation & Key Concepts

### Q1: What is the difference between TF-IDF and Transformer embeddings?
- **TF-IDF (Term Frequency-Inverse Document Frequency):** A classical statistical weighting method representing documents as sparse bag-of-words vectors. It reflects how distinct a word is to a document relative to a corpus, but ignores word order, semantics, and polysemy.
- **Transformers (RoBERTa / BERT):** Deep neural networks using multi-head self-attention to generate dense, contextualized embeddings where the representation of each token depends dynamically on its surrounding sentence context.

### Q2: Why is dual-stream preprocessing necessary?
- Emojis and capitalization contain crucial sentiment signals (e.g. `😍` signifies affection, all-caps `TERRIBLE` signifies high intensity). Stripping punctuation and emojis would degrade transformer performance.
- Conversely, classical TF-IDF benefits from case-folding, stop-word elimination, and lemmatization to avoid sparse vector fragmentation.

### Q3: How is mixed sentiment handled?
- When a text contains contrastive conjunctions (`but`, `however`) joining opposing lexical cues, or when model softmax probabilities are closely split between positive and negative classes, the system raises a dedicated **Mixed Signal Alert** detailing the conflicting components.

---

## 🔒 10. Privacy & Security

- **In-Memory Processing:** All text entered into the application is analyzed in volatile RAM and is never stored permanently on disk or transmitted to external third parties.
- **No Secret Hardcoding:** Compliant with security best practices; models run locally via Hugging Face cache.

---

## 📜 11. License
Academic Open-Source DRTM / Text Mining Project.
#   S e n t i m e n t - E m o t i o n s - A n a l i z e r  
 