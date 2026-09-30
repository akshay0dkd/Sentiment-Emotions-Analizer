"""
Intelligent Sentiment and Emotion Analysis Application.
Built with Streamlit, NLTK, scikit-learn, and Hugging Face Transformers.
"""

import streamlit as st
import pandas as pd
import json
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from modules.preprocessing import TextPreprocessor
from modules.feature_extraction import FeatureExtractor
from modules.tfidf_analyzer import TfidfAnalyzer
from modules.emoji_analyzer import EmojiAnalyzer
from modules.sentiment_analyzer import SentimentAnalyzer
from modules.emotion_analyzer import EmotionAnalyzer
from modules.visualization import (
    create_sentiment_chart,
    create_emotion_donut_chart,
    create_emotion_bar_chart,
    create_tfidf_chart
)
from models.model_loader import ModelRegistry, MODEL_METADATA, get_device

# Configure Streamlit Page
st.set_page_config(
    page_title="Intelligent Sentiment & Emotion Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        padding: 24px 30px;
        border-radius: 14px;
        color: #FFFFFF;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: #FFFFFF !important;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
        padding-bottom: 6px;
    }
    .main-header p {
        color: #C7D2FE !important;
        font-size: 15px;
        margin: 0;
    }
    
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 14px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
    }
    .metric-card-title {
        font-size: 13px;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .metric-card-val {
        font-size: 24px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 4px;
    }
    .metric-card-sub {
        font-size: 12px;
        color: #4B5563;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-pos { background-color: #D1FAE5; color: #065F46; }
    .badge-neg { background-color: #FEE2E2; color: #991B1B; }
    .badge-neu { background-color: #E5E7EB; color: #374151; }
    .badge-mixed { background-color: #FEF3C7; color: #92400E; }
    
    .token-box {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        padding: 8px 12px;
        border-radius: 8px;
        font-family: monospace;
        font-size: 13px;
        color: #374151;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if 'history' not in st.session_state:
    st.session_state.history = []
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""
if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = None


# Cache pipeline loaders via Streamlit
@st.cache_resource(show_spinner="Loading NLP & Transformer Models into Memory...")
def load_all_components():
    """Initializes and caches pipeline models and feature extractors."""
    preprocessor = TextPreprocessor()
    feature_extractor = FeatureExtractor()
    tfidf_analyzer = TfidfAnalyzer()
    emoji_analyzer = EmojiAnalyzer()
    
    # Pre-warm transformer pipelines into RAM
    ModelRegistry.load_sentiment_pipeline()
    ModelRegistry.load_emotion_pipeline()
    
    sentiment_analyzer = SentimentAnalyzer(ModelRegistry.load_sentiment_pipeline)
    emotion_analyzer = EmotionAnalyzer(ModelRegistry.load_emotion_pipeline)
    return preprocessor, feature_extractor, tfidf_analyzer, emoji_analyzer, sentiment_analyzer, emotion_analyzer


# Load sample test cases
@st.cache_data
def load_sample_test_cases():
    filepath = os.path.join(os.path.dirname(__file__), "data", "sample_data", "test_cases.json")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# Load cached engines
preprocessor, feature_extractor, tfidf_analyzer, emoji_analyzer, sentiment_analyzer, emotion_analyzer = load_all_components()
test_cases = load_sample_test_cases()


# SIDEBAR CONFIGURATION
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/artificial-intelligence.png", width=64)
    st.title("System Specs & Models")
    
    dev_id, dev_name = get_device()
    st.info(f"⚡ **Compute Hardware:** `{dev_name}`")
    
    with st.expander("🤖 **Sentiment Model Architecture**", expanded=False):
        meta_s = MODEL_METADATA["sentiment"]
        st.markdown(f"**Model:** `{meta_s['name']}`")
        st.markdown(f"**HuggingFace ID:** `{meta_s['huggingface_id']}`")
        st.markdown(f"**Architecture:** {meta_s['architecture']}")
        st.markdown(f"**Domain:** {meta_s['domain']}")
        st.markdown(f"**Classes:** `Positive`, `Neutral`, `Negative`")

    with st.expander("🎭 **Emotion Model Architecture**", expanded=False):
        meta_e = MODEL_METADATA["emotion"]
        st.markdown(f"**Model:** `{meta_e['name']}`")
        st.markdown(f"**HuggingFace ID:** `{meta_e['huggingface_id']}`")
        st.markdown(f"**Architecture:** {meta_e['architecture']}")
        st.markdown(f"**Taxonomy:** Ekman (Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral)")

    st.markdown("---")
    st.subheader("📚 Quick Sample Presets")
    selected_sample = st.selectbox(
        "Select an academic benchmark scenario:",
        options=["-- Select a sample prompt --"] + [f"{c['category']}: {c['label']}" for c in test_cases],
        index=0
    )
    if selected_sample != "-- Select a sample prompt --":
        idx = [f"{c['category']}: {c['label']}" for c in test_cases].index(selected_sample)
        st.session_state.input_text = test_cases[idx]['text']

    st.markdown("---")
    st.subheader("📜 Session Analysis History")
    if st.session_state.history:
        st.write(f"Total analyses: **{len(st.session_state.history)}**")
        for i, item in enumerate(reversed(st.session_state.history[-5:])):
            st.caption(f"**{len(st.session_state.history)-i}.** *\"{item['preview']}\"*")
            st.markdown(f"`{item['sentiment']} ({item['sentiment_conf']}%)` • `{item['emotion']}`")
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No text analyzed in this session yet.")

    st.markdown("---")
    st.caption("🔒 **Privacy Guarantee:** All text entered is analyzed in volatile memory and is never permanently stored or shared.")


# MAIN APPLICATION HEADER
st.markdown("""
<div class="main-header">
    <h1>🧠 Intelligent Sentiment & Emotion Analyzer</h1>
    <p>Comprehensive Text Mining, NLP Preprocessing, TF-IDF Vectorization, Emoji Extraction & Pretrained Hugging Face Transformers</p>
</div>
""", unsafe_allow_html=True)


# INPUT AREA
st.markdown("### ✍️ Input Text for Analysis")

# Text Area with dynamic key bound to session state
text_input = st.text_area(
    label="Enter or paste any sentence, customer review, or social media post below:",
    value=st.session_state.input_text,
    height=125,
    placeholder="Type something here (e.g. 'I absolutely love this product! It exceeded all expectations 😍❤️🔥' or 'Great, another two hour delay... wonderful service 🙃')",
    help="Preserves emojis, handles punctuation, and supports multi-sentence inputs."
)

col_btn1, col_btn2, col_btn_spacer = st.columns([1.5, 1.2, 5])
with col_btn1:
    analyze_clicked = st.button("🔍 **Analyze Text**", type="primary", use_container_width=True)
with col_btn2:
    if st.button("🗑️ **Clear**", use_container_width=True):
        st.session_state.input_text = ""
        st.session_state.last_analysis = None
        st.rerun()


# ANALYSIS EXECUTION
if analyze_clicked:
    if not text_input or not text_input.strip():
        st.warning("⚠️ Please enter valid text before clicking Analyze.")
    else:
        with st.spinner("Processing NLP Pipeline, Feature Extraction, and Transformer Inference..."):
            try:
                # 1. Text Preprocessing (Dual-Stream)
                preproc_data = preprocessor.process_for_text_mining(text_input)
                
                # 2. Emoji Analysis
                emoji_data = emoji_analyzer.analyze_emojis(text_input)
                
                # 3. Feature Extraction
                features_data = feature_extractor.extract_features(
                    text=text_input,
                    preprocessed_data=preproc_data,
                    emoji_count=emoji_data['total_emoji_count']
                )
                
                # 4. TF-IDF Text Mining Analysis
                tfidf_terms = tfidf_analyzer.extract_top_terms(preproc_data['processed_text'])
                
                # 5. Transformer Sentiment Inference
                sentiment_data = sentiment_analyzer.analyze(
                    text=preproc_data['model_input_text'],
                    feature_data=features_data
                )
                
                # 6. Transformer Emotion Classification
                emotion_data = emotion_analyzer.analyze(
                    text=preproc_data['model_input_text']
                )
                
                # Save into session state
                results = {
                    'raw_text': text_input,
                    'preproc': preproc_data,
                    'emoji': emoji_data,
                    'features': features_data,
                    'tfidf': tfidf_terms,
                    'sentiment': sentiment_data,
                    'emotion': emotion_data
                }
                st.session_state.last_analysis = results
                
                # Append to history
                preview = (text_input[:45] + "...") if len(text_input) > 45 else text_input
                st.session_state.history.append({
                    'preview': preview,
                    'sentiment': sentiment_data['dominant_sentiment'],
                    'sentiment_conf': sentiment_data['confidence'],
                    'emotion': emotion_data['dominant_emotion']
                })
                
            except Exception as e:
                st.error(f"❌ An error occurred during analysis: {str(e)}")


# DISPLAY RESULTS DASHBOARD
if st.session_state.last_analysis:
    res = st.session_state.last_analysis
    st.markdown("---")
    st.subheader("📊 Executive Analysis Summary")

    # 4 Metric Executive Summary Cards
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        s_label = res['sentiment']['dominant_sentiment']
        s_conf = res['sentiment']['confidence']
        s_color = res['sentiment']['color']
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {s_color};">
            <div class="metric-card-title">Overall Sentiment</div>
            <div class="metric-card-val" style="color:{s_color};">{res['sentiment']['display_label']}</div>
            <div class="metric-card-sub">Confidence: <b>{s_conf:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        e_label = res['emotion']['dominant_emotion']
        e_conf = res['emotion']['confidence']
        e_color = res['emotion']['color']
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {e_color};">
            <div class="metric-card-title">Dominant Emotion</div>
            <div class="metric-card-val" style="color:{e_color};">{res['emotion']['display_label']}</div>
            <div class="metric-card-sub">Model Probability: <b>{e_conf:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        em_count = res['emoji']['total_emoji_count']
        em_str = res['emoji']['emoji_string'] if em_count > 0 else "None"
        em_desc = res['emoji']['emoji_sentiment_summary']
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid #F59E0B;">
            <div class="metric-card-title">Detected Emojis ({em_count})</div>
            <div class="metric-card-val" style="font-size:20px;">{em_str[:12]}</div>
            <div class="metric-card-sub">Polarity: <b>{em_desc}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        stats = res['features']['statistics']
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid #6366F1;">
            <div class="metric-card-title">Text Metrics</div>
            <div class="metric-card-val">{stats['word_count']} <span style="font-size:14px;font-weight:normal;color:#6B7280;">words</span></div>
            <div class="metric-card-sub">{stats['character_count']} chars • {stats['sentence_count']} sent • {stats['emoji_count']} emojis</div>
        </div>
        """, unsafe_allow_html=True)

    # Mixed / Contrastive Signal Alert if detected
    mixed = res['sentiment']['mixed_analysis']
    if mixed.get('is_mixed'):
        st.warning(f"⚠️ **Mixed / Contrastive Signal Detected:** {mixed.get('explanation')}")
        if mixed.get('positive_indicators') or mixed.get('negative_indicators'):
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown(f"**Positive Indicators:** `{', '.join(mixed.get('positive_indicators', [])) or 'None'}`")
            with col_m2:
                st.markdown(f"**Negative Indicators:** `{', '.join(mixed.get('negative_indicators', [])) or 'None'}`")

    # DETAIL TABS
    tab_viz, tab_mining, tab_linguistic, tab_emojis, tab_theory = st.tabs([
        "📈 Visual Dashboard",
        "📚 Text Mining & TF-IDF",
        "🔎 Linguistic & Preprocessing",
        "😀 Emoji Semantics",
        "🤖 How The Models Work"
    ])

    # TAB 1: Visual Dashboard
    with tab_viz:
        col_v1, col_v2 = st.columns([1, 1])
        with col_v1:
            st.plotly_chart(create_sentiment_chart(res['sentiment']), use_container_width=True)
            
            # Probability detail table
            st.caption("Detailed Sentiment Probabilities:")
            st.dataframe(
                pd.DataFrame([
                    {"Class": k, "Score": v, "Percentage": f"{res['sentiment']['probabilities_pct'][k]}%"}
                    for k, v in res['sentiment']['probabilities'].items()
                ]),
                use_container_width=True,
                hide_index=True
            )
            
        with col_v2:
            st.plotly_chart(create_emotion_donut_chart(res['emotion']), use_container_width=True)
            st.plotly_chart(create_emotion_bar_chart(res['emotion']), use_container_width=True)

    # TAB 2: Text Mining & TF-IDF
    with tab_mining:
        st.markdown("#### 📚 Classical Text Mining: Term Frequency - Inverse Document Frequency (TF-IDF)")
        st.markdown(
            "TF-IDF measures how important a term is to a document relative to a corpus: "
            r"$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{|D|}{1 + |\{d \in D : t \in d\}|}\right)$."
        )
        
        col_t1, col_t2 = st.columns([1.2, 1])
        with col_t1:
            st.plotly_chart(create_tfidf_chart(res['tfidf']), use_container_width=True)
            
        with col_t2:
            st.markdown("**Ranked TF-IDF Terms**")
            if res['tfidf']:
                st.dataframe(
                    pd.DataFrame(res['tfidf'])[['term', 'score', 'percentage']],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No distinct vocabulary terms extracted for TF-IDF ranking.")

        st.markdown("---")
        st.markdown("#### 📊 Surface & Lexical Text Features")
        f_surf = res['features']['surface_counts']
        f_stat = res['features']['statistics']
        f_lex = res['features']['lexical_cues']
        
        sf_col1, sf_col2, sf_col3 = st.columns(3)
        with sf_col1:
            st.markdown(f"- **Total Characters:** `{f_stat['character_count']}`")
            st.markdown(f"- **Characters (No spaces):** `{f_stat['character_no_spaces']}`")
            st.markdown(f"- **Average Word Length:** `{f_stat['avg_word_length']}` chars")
        with sf_col2:
            st.markdown(f"- **Uppercase Words (SHOUTING):** `{f_surf['uppercase_words']}` ({f_surf['uppercase_ratio_pct']}%)")
            st.markdown(f"- **Exclamations (!):** `{f_surf['exclamation_count']}`")
            st.markdown(f"- **Question Marks (?):** `{f_surf['question_count']}`")
        with sf_col3:
            st.markdown(f"- **Hashtags (#):** `{f_surf['hashtag_count']}`")
            st.markdown(f"- **User Mentions (@):** `{f_surf['mention_count']}`")
            st.markdown(f"- **URLs:** `{f_surf['url_count']}`")

    # TAB 3: Linguistic & Preprocessing
    with tab_linguistic:
        st.markdown("#### 🔎 Dual-Stream Preprocessing Inspection")
        st.caption("Demonstrating the difference between Neural Transformer input (preserving contextual cues and emojis) and Classical Linguistic Mining input.")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("**1. Model Stream (Transformer Ready)**")
            st.markdown(f"<div class='token-box'>{res['preproc']['model_input_text']}</div>", unsafe_allow_html=True)
            st.caption("Preserves emojis, casing, and sentence punctuation for self-attention context.")
            
        with col_p2:
            st.markdown("**2. Linguistic Stream (Text Mining & TF-IDF)**")
            st.markdown(f"<div class='token-box'>{res['preproc']['processed_text'] or '[Empty after stop-word removal]'}</div>", unsafe_allow_html=True)
            st.caption("Lowercased, punctuation-stripped, stop-words removed, and lemmatized.")

        st.markdown("---")
        st.markdown("#### 🧩 Tokenization & Lemmatization Breakdown")
        
        col_tok1, col_tok2 = st.columns(2)
        with col_tok1:
            st.markdown("**Raw Tokens:**")
            st.write(res['preproc']['raw_tokens'])
            
            st.markdown("**Stop Words Identified & Filtered:**")
            if res['preproc']['stop_words_found']:
                st.write(list(set(res['preproc']['stop_words_found'])))
            else:
                st.caption("No standard English stop words found.")
                
        with col_tok2:
            st.markdown("**Lemmas (WordNet Lemmatization):**")
            st.write(res['preproc']['lemmas'])
            
            meta = res['preproc']['metadata']
            if meta['urls'] or meta['mentions'] or meta['hashtags']:
                st.markdown("**Extracted Entities & Metadata:**")
                st.json(meta)

    # TAB 4: Emoji Semantics
    with tab_emojis:
        st.markdown("#### 😀 Emoji Detection & Semantic Interpretation")
        st.caption("Emojis provide vital non-verbal emotional cues. The system parses them independently while maintaining them in the Transformer token stream.")
        
        if res['emoji']['total_emoji_count'] > 0:
            st.success(f"Detected **{res['emoji']['total_emoji_count']}** total emoji(s) across **{res['emoji']['unique_emoji_count']}** unique character(s).")
            st.dataframe(
                pd.DataFrame(res['emoji']['emoji_breakdown']),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No Unicode emojis were detected in the input text.")

    # TAB 5: How The Models Work (Academic Theory)
    with tab_theory:
        st.markdown("### 🤖 Architecture & Theoretical Foundations")
        
        st.markdown("""
        #### 1. Classical Text Mining (TF-IDF) vs Modern Deep Learning (Transformers)
        | Dimension | Classical Text Mining (TF-IDF + ML) | Transformer NLP (RoBERTa / BERT) |
        | :--- | :--- | :--- |
        | **Representation** | Sparse Bag-of-Words vectors | Dense contextual embedding vectors |
        | **Word Order** | Ignored (Order-invariant) | Retained via Positional Encodings |
        | **Polysemy** | Cannot differentiate context | Dynamic contextual representation |
        | **Computation** | Fast, linear time, CPU friendly | Self-attention matrix multiplication ($O(N^2)$) |
        """)

        st.markdown("""
        #### 2. End-to-End Pipeline Workflow
        ```text
           User Input Text
                 ↓
          [Unicode Normalization (NFKC)]
                 ↓
        ┌────────────────────────────────────────────────────────┐
        │                                                        │
        │  Model Stream                     Linguistic Stream    │
        │  (Emojis + Case + Punctuation)    (Stop-words filtered)│
        │         ↓                                 ↓            │
        │  Transformer Tokenizer             WordNet Lemmatizer  │
        │         ↓                                 ↓            │
        │  [RoBERTa / DistilRoBERTa]         [TF-IDF Vectorizer] │
        │         ↓                                 ↓            │
        │  Softmax Probabilities             Keyword Weights     │
        └───────────────────────┬────────────────────────────────┘
                                ↓
                 [Result Aggregator & Plotly Dashboard]
        ```
        """)

        st.markdown("""
        #### 3. Key Limitations to Note 
        - **Sarcasm & Irony:** Sarcasm often uses positive words in negative contexts without overt lexical cues.
        - **Cross-Lingual Code-Switching:** English-specific models may misclassify multilingual idioms.
        - **Contrastive Aspects:** Sentences like *"The camera is great but the battery is terrible"* span multiple aspects and require aspect-based sentiment breakdown.
        """)

# FOOTER
st.markdown("---")
st.markdown(
    "<center><small style='color: #6B7280;'>"
    "Intelligent Sentiment & Emotion Analysis • Academic Project (DRTM/Text Mining) • Streamlit + PyTorch + Hugging Face"
    "</small></center>",
    unsafe_allow_html=True
)
