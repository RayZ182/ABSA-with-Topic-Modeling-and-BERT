import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim import corpora
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Download NLTK data (if not already done)
# nltk.download('punkt')
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
negation_words = {'not', 'no', 'never'}
stop_words = stop_words - negation_words  # Remove negation words from stopwords

# Load the saved LDA model and dictionary
lda_model = gensim.models.LdaModel.load("src/lda_model.gensim")
dictionary = corpora.Dictionary.load("src/dictionary.gensim")

# Define topic-to-aspect mapping (same as in your original pipeline)
topic_to_aspect = {
    0: "general_viewing_experience",
    1: "story_and_performance",
    2: "production_quality",
    3: "narrative_tone",
    4: "character_roles_and_execution",
    -1: "unknown"
}

# Define keyword dictionary (same as in your original pipeline)
keyword_aspects = {
    "acting": ["actor", "actress", "performance", "cast", "role", "played", "plays"],
    "plot": ["story", "plot", "narrative", "script", "ending"],
    "cinematography": ["visual", "cinematography", "camera", "shot", "scene", "scenes", "look"],
    "soundtrack": ["music", "sound", "score", "soundtrack", "audio"],
    "direction": ["director", "direction", "paced", "style", "vision"],
    "overall": ["movie", "film", "good", "bad", "great"]
}

# Load BERT model and tokenizer
model_name = "textattack/bert-base-uncased-SST-2"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Preprocessing function
def preprocess_review(review):
    review = review.lower().replace("br", "")
    sentences = sent_tokenize(review)
    cleaned_sentences = []
    for sent in sentences:
        words = word_tokenize(sent)
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

# Interactive tool function
def analyze_new_review(review):
    # Preprocess the review
    cleaned_sentences = preprocess_review(review)
    
    # Assign aspects
    aspect_sentiment_pairs = []
    for sent in cleaned_sentences:
        aspect = assign_aspect_with_keywords(sent)
        sentiment = get_sentiment(sent)
        aspect_sentiment_pairs.append((aspect, sentiment))
    
    return aspect_sentiment_pairs

# Main interactive loop
def main():
    print("Welcome to the ABSA Interactive Tool!")
    print("Enter a movie review (or type 'exit' to quit):")
    while True:
        review = input("> ")
        if review.lower() == "exit":
            print("Exiting the tool. Goodbye!")
            break
        if not review.strip():
            print("Please enter a non-empty review.")
            continue
        
        # Analyze the review
        results = analyze_new_review(review)
        
        # Display results
        print("\nAspect-Sentiment Pairs:")
        for aspect, sentiment in results:
            print(f"- {aspect}: {sentiment}")
        print("\nEnter another review (or type 'exit' to quit):")

if __name__ == "__main__":
    main()