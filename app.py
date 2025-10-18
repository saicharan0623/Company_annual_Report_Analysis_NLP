import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import fitz  # PyMuPDF
import warnings
import nltk

# --- Page Configuration ---
st.set_page_config(
    page_title="L&T Annual Report NLP Analyzer",
    page_icon="📄",
    layout="wide"
)

warnings.filterwarnings("ignore")

# Safe NLTK downloads
for resource in ['stopwords', 'punkt']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# --- Caching Functions ---
@st.cache_data
def load_and_process_pdf(file_path):
    """Load PDF, extract text, clean, and compute sentiment."""
    doc = fitz.open(file_path)
    pages_data = [{'page_num': i+1, 'text': page.get_text("text")}
                  for i, page in enumerate(doc) if page.get_text("text").strip()]
    df = pd.DataFrame(pages_data)

    stop_words = set(nltk.corpus.stopwords.words('english'))
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        filtered = [w for w in tokens if w not in stop_words and len(w) > 2]
        return " ".join(filtered)
    df['cleaned_text'] = df['text'].apply(preprocess_text)

    # Sentiment analysis
    df['sentiment'] = df['text'].apply(lambda t: TextBlob(t).sentiment.polarity)
    return df

# --- Main App ---
st.title("📄 NLP Analysis of L&T Annual Report")
st.markdown("Dashboard displaying sentiment, word cloud, and topic modeling.")

PDF_PATH = "LT_Annual_Report.pdf"  # Make sure this exists

try:
    with st.spinner("Processing PDF..."):
        df = load_and_process_pdf(PDF_PATH)
        full_text = " ".join(df['text'])
        full_cleaned_text = " ".join(df['cleaned_text'])

    st.success("Analysis complete!")

    # --- Sentiment ---
    st.header("Overall Report Sentiment")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Trend Across Pages")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df['page_num'], df['sentiment'], color='royalblue')
        ax.axhline(0, color='red', linestyle='--')
        ax.set_xlabel("Page Number")
        ax.set_ylabel("Sentiment")
        st.pyplot(fig)

    with col2:
        st.subheader("Sentiment Distribution")
        fig2, ax2 = plt.subplots(figsize=(10,5))
        sns.histplot(df['sentiment'], bins=20, kde=True, color='skyblue', ax=ax2)
        st.pyplot(fig2)

    st.markdown("---")
    st.header("Top 30 Most Frequent Words")
    word_counts = Counter(full_cleaned_text.split())
    top30 = pd.DataFrame(word_counts.most_common(30), columns=['Word', 'Frequency'])
    st.dataframe(top30, height=300)

    st.header("Word Cloud")
    wordcloud = WordCloud(width=800, height=600, background_color='white').generate(full_cleaned_text)
    fig_wc, ax_wc = plt.subplots(figsize=(10,7))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)

    st.markdown("---")
    st.header("Topic Modeling (LDA)")
    with st.spinner("Building topic model..."):
        num_topics = 5
        tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, stop_words='english')
        dtm_tfidf = tfidf_vectorizer.fit_transform(df['cleaned_text'])
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda_model.fit(dtm_tfidf)

        st.subheader(f"Top {num_topics} Topics")
        feature_names = tfidf_vectorizer.get_feature_names_out()
        for idx, topic in enumerate(lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10-1:-1]]
            st.write(f"**Topic {idx+1}:** {', '.join(top_words)}")

except FileNotFoundError:
    st.error(f"Error: File '{PDF_PATH}' not found.")
except Exception as e:
    st.error(f"An error occurred: {e}")
