import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim import corpora
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

# Download NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# Preserve negation words
negation_words = {'not', 'no', 'never'}
stop_words = stop_words - negation_words 

# Define paths to LDA model and dictionary
LDA_MODEL_PATH = "lda_model.gensim"
DICTIONARY_PATH = "dictionary.gensim"
try:
    lda_model = gensim.models.LdaModel.load(LDA_MODEL_PATH)
    dictionary = corpora.Dictionary.load(DICTIONARY_PATH)
except Exception as e:
    st.error(f"Error loading LDA model or dictionary: {str(e)}")
    st.stop()

# Define topic-to-aspect mapping
topic_to_aspect = {
    0: "general_viewing_experience",
    1: "story_and_performance",
    2: "production_quality",
    3: "narrative_tone",
    4: "character_roles_and_execution",
    -1: "unknown"
}

# Define keyword dictionary
keyword_aspects = {
    "acting": ["actor", "actress", "performance", "cast", "role", "played", "plays", "acting"],
    "plot": ["story", "plot", "narrative", "script", "ending"],
    "cinematography": ["visual", "cinematography", "camera", "shot", "scene", "scenes", "look"],
    "soundtrack": ["music", "sound", "score", "soundtrack", "audio", "songs"],
    "direction": ["director", "direction", "paced", "style", "vision"],
    "overall": ["movie", "film", "good", "bad", "great"]
}

# Load BERT model and tokenizer
model_name = "textattack/bert-base-uncased-SST-2"
try:
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
except Exception as e:
    st.error(f"Error loading BERT model: {str(e)}")
    st.stop()

# Preprocessing function
def preprocess_review(review):
    review = review.lower().replace("br", "")
    sentences = sent_tokenize(review)
    cleaned_sentences = []
    for sent in sentences:
        sub_sentences = [s.strip() for s in sent.split(",")]
        for sub_sent in sub_sentences:
            if sub_sent:
                words = word_tokenize(sub_sent)
                words = [word for word in words if word.isalnum() and word not in stop_words]
                if words:
                    cleaned_sentences.append(" ".join(words))
    return cleaned_sentences

# Function to assign aspects using LDA and keyword fallback
def get_dominant_topic(sentence):
    if not sentence:
        return -1
    bow = dictionary.doc2bow(sentence.split())
    topics = lda_model[bow]
    return max(topics, key=lambda x: x[1])[0] if topics else -1

def assign_aspect_with_keywords(sentence):
    for aspect, keywords in keyword_aspects.items():
        if any(kw in sentence for kw in keywords):
            return aspect
    return topic_to_aspect[get_dominant_topic(sentence)]

# Function to predict sentiment using BERT
def get_sentiment(text):
    if not text:
        return "positive"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    return "positive" if torch.argmax(logits, dim=1).item() == 1 else "negative"

# Function to analyze a review
def analyze_new_review(review):
    cleaned_sentences = preprocess_review(review)
    aspect_sentiment_pairs = []
    for sent in cleaned_sentences:
        aspect = assign_aspect_with_keywords(sent)
        sentiment = get_sentiment(sent)
        aspect_sentiment_pairs.append((aspect, sentiment))
    return aspect_sentiment_pairs

# Streamlit app layout
st.title("Aspect-Based Sentiment Analysis for Movie Reviews")
st.write("Enter a movie review below to analyze its aspects and sentiments.")

# Text input for the review
review = st.text_area("Movie Review", placeholder="e.g., The acting was great, but the plot was not good.")

# Analyze button
if st.button("Analyze"):
    if not review.strip():
        st.warning("Please enter a non-empty review.")
    else:
        try:
            with st.spinner("Analyzing your review..."):
                results = analyze_new_review(review)
            st.success("Analysis complete!")
            st.subheader("Aspect-Sentiment Pairs:")
            for aspect, sentiment in results:
                st.write(f"- **{aspect}**: {sentiment}")
        except Exception as e:
            st.error(f"Error analyzing review: {str(e)}")

# Footer
st.write("Built with Streamlit | ABSA Project")