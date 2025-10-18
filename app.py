import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pyLDAvis
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import fitz  # PyMuPDF
import warnings
import subprocess
import sys

# --- Page Configuration ---
st.set_page_config(
    page_title="L&T Annual Report NLP Analyzer",
    page_icon="📄",
    layout="wide"
)

# --- Suppress Warnings ---
warnings.filterwarnings('ignore')

# --- Safe NLTK downloads ---
import nltk
for resource in ['stopwords', 'punkt']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# --- Caching Functions ---
@st.cache_data
def load_and_process_pdf(file_path):
    """Load PDF, extract text, clean, and perform sentiment analysis."""
    doc = fitz.open(file_path)
    pages_data = [{'page_num': page_num + 1, 'text': page.get_text("text")}
                  for page_num, page in enumerate(doc) if page.get_text("text").strip()]
    df = pd.DataFrame(pages_data)

    # Text preprocessing
    stop_words = set(nltk.corpus.stopwords.words('english'))
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        return " ".join(filtered_tokens)
    df['cleaned_text'] = df['text'].apply(preprocess_text)

    # Sentiment analysis
    df['sentiment'] = df['text'].apply(lambda text: TextBlob(text).sentiment.polarity)

    return df

@st.cache_resource
def load_spacy_model():
    """Lazy-load spaCy model for NER."""
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # If model is missing, download it at runtime
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

# --- Main App UI ---
st.title("📄 NLP Analysis of the L&T Annual Report")
st.markdown("This dashboard displays a detailed NLP analysis of the L&T annual report.")

# PDF file path
PDF_PATH = "LT_Annual_Report.pdf"  # Ensure this file exists

try:
    with st.spinner("Analyzing document... This might take a minute."):
        df = load_and_process_pdf(PDF_PATH)
        full_text = " ".join(df['text'])
        full_cleaned_text = " ".join(df['cleaned_text'])

    st.success("Analysis complete! Here are the insights:")

    # --- Sentiment Visualizations ---
    st.header("Overall Report Sentiment")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Trend Across Pages")
        fig_trend, ax_trend = plt.subplots(figsize=(10, 5))
        ax_trend.plot(df['page_num'], df['sentiment'], color='royalblue', alpha=0.8)
        ax_trend.axhline(y=0, color='r', linestyle='--')
        ax_trend.set_title('Sentiment Trend')
        ax_trend.set_xlabel('Page Number')
        ax_trend.set_ylabel('Sentiment Polarity')
        ax_trend.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_trend)

    with col2:
        st.subheader("Distribution of Sentiment Scores")
        fig_dist, ax_dist = plt.subplots(figsize=(10, 5))
        sns.histplot(df['sentiment'], bins=20, kde=True, color='skyblue', ax=ax_dist)
        ax_dist.set_title("Sentiment Score Distribution")
        ax_dist.set_xlabel("Sentiment Score")
        ax_dist.set_ylabel("Number of Pages")
        st.pyplot(fig_dist)

    st.markdown("---")
    st.header("Key Content Analysis")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Top Mentioned Entities")
        nlp = load_spacy_model()
        doc = nlp(full_text[:1000000])  # Limit to 1M chars
        entities = {"PERSON": [], "GPE": [], "ORG": []}
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text.strip())
        for label, items in entities.items():
            count = Counter(items)
            st.markdown(f"**{label}:**")
            for item, freq in count.most_common(5):
                st.write(f"  - {item}: {freq} mentions")

        st.subheader("Top 30 Most Frequent Words")
        all_words = full_cleaned_text.split()
        word_counts = Counter(all_words)
        top_30_df = pd.DataFrame(word_counts.most_common(30), columns=['Word', 'Frequency'])
        st.dataframe(top_30_df, height=300)

    with col4:
        st.subheader("Word Cloud Visualization")
        wordcloud = WordCloud(width=800, height=600, background_color='white').generate(full_cleaned_text)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 7))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)

    st.markdown("---")
    st.header("Topic Modeling with LDA")

    with st.spinner("Building topic model..."):
        num_topics = 5
        tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=5, stop_words='english')
        dtm_tfidf = tfidf_vectorizer.fit_transform(df['cleaned_text'])
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda_model.fit(dtm_tfidf)

        st.subheader(f"Top {num_topics} Discovered Topics")
        feature_names = tfidf_vectorizer.get_feature_names_out()
        for idx, topic in enumerate(lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
            st.write(f"**Topic {idx+1}:** {', '.join(top_words)}")

        st.subheader("Average Sentiment per Topic")
        topic_distribution = lda_model.transform(dtm_tfidf)
        df['topic'] = topic_distribution.argmax(axis=1)
        sentiment_by_topic = df.groupby('topic')['sentiment'].mean().sort_values(ascending=False)
        st.dataframe(sentiment_by_topic)

    with st.expander("Explore Interactive Topic Model (pyLDAvis)"):
        with st.spinner("Preparing interactive visualization..."):
            count_vectorizer = CountVectorizer(vocabulary=tfidf_vectorizer.get_feature_names_out())
            dtm_counts = count_vectorizer.fit_transform(df['cleaned_text'])
            vis_data = pyLDAvis.prepare(
                topic_term_dists=lda_model.components_,
                doc_topic_dists=topic_distribution,
                doc_lengths=df['cleaned_text'].apply(lambda x: len(x.split())).tolist(),
                vocab=tfidf_vectorizer.get_feature_names_out(),
                term_frequency=dtm_counts.sum(axis=0).A1,
                mds='tsne'
            )
            html_string = pyLDAvis.prepared_data_to_html(vis_data)
            st.components.v1.html(html_string, width=1300, height=800, scrolling=True)

except FileNotFoundError:
    st.error(f"Error: The file '{PDF_PATH}' was not found. Make sure it's in the same folder as app.py.")
except Exception as e:
    st.error(f"An error occurred during analysis: {e}")
