import pandas as pd
import nltk
import string


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load dataset
data = pd.read_csv(r"c:\Users\user\Desktop\maryam\full_faq_dataset.csv")

# Text preprocessing function
def preprocess(text):

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))

    filtered_words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(filtered_words)

# Preprocess all questions
data['processed_question'] = data['question'].apply(preprocess)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(
    data['processed_question']
)

# Chatbot function
def chatbot(user_query):

    # Preprocess user query
    processed_query = preprocess(user_query)

    # Convert query to vector
    query_vector = vectorizer.transform([processed_query])

    # Compute similarity
    similarity_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    )

    # Best match index
    best_match_index = similarity_scores.argmax()

    # Best score
    best_score = similarity_scores[0, best_match_index]

    # Confidence threshold
    threshold = 0.3

    if best_score < threshold:
        return (
            "Sorry, I could not find a reliable answer.",
            best_score
        )

    answer = data.iloc[best_match_index]['answer']

    return answer, best_score

# Main loop
print("FAQ Chatbot Started")
print("Type 'exit' to quit")

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Chatbot ended.")
        break

    response, score = chatbot(user_input)

    print(f"Bot: {response}")
    print(f"Similarity Score: {score:.2f}")