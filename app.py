import streamlit as st
import fitz
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Textbook AI Search", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("📚 Navigation")
subject = st.sidebar.radio("Go to:", ["Science Textbooks", "Social Textbooks"])

# --- DATA LOADING FUNCTION ---
@st.cache_resource
def load_data(folder_name):
    corpus = []
    # Note: On the cloud, we use relative paths (no D:\)
    if os.path.exists(folder_name):
        for filename in os.listdir(folder_name):
            if filename.endswith(".pdf"):
                try:
                    doc = fitz.open(os.path.join(folder_name, filename))
                    for page in doc:
                        text = page.get_text().split('\n\n')
                        corpus.extend([p.strip() for p in text if len(p.strip()) > 50])
                except Exception as e:
                    st.error(f"Error reading {filename}: {e}")
    return corpus

# --- SUBJECT LOGIC ---
if subject == "Science Textbooks":
    st.title("🔬 Science AI Search")
    folder = "nlpproject"  # This folder must be in your GitHub
elif subject == "Social Textbooks":
    st.title("🌍 Social AI Search")
    folder = "nlpproject1" # This folder must be in your GitHub

all_text_data = load_data(folder)

# --- SEARCH INTERFACE ---
if all_text_data:
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_text_data)

    user_query = st.text_input(f"Search in {subject}:", placeholder="Enter keywords...")

    if user_query:
        query_vec = vectorizer.transform([user_query])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        best_idx = scores.argmax()

        if scores[best_idx] > 0.1:
            st.success(f"Match Found! (Similarity: {scores[best_idx]:.2f})")
            st.info(all_text_data[best_idx])
        else:
            st.warning("No close match found. Try different keywords.")
else:
    st.error(f"No PDFs found in the '{folder}' folder. Please check your GitHub repository.")