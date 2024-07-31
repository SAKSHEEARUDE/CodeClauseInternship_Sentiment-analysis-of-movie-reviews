import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Sample movie review data
data = {
    'review': [
        'I loved this movie. It was fantastic!',
        'The movie was okay, not great but not terrible.',
        'I hated the movie. It was awful.',
        'It was a boring film with no plot.',
        'What an amazing movie! I would watch it again!',
        'The film was mediocre and a waste of time.',
        'Fantastic! A must-see for everyone.',
        'The plot was predictable and boring.'
    ],
    'sentiment': ['positive', 'neutral', 'negative', 'negative', 'positive', 'negative', 'positive', 'negative']
}

# Create DataFrame
df = pd.DataFrame(data)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)

# Apply preprocessing
df['review'] = df['review'].apply(preprocess_text)

# Split data into features and labels
X = df['review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model training
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Predict function
def predict_sentiment(review):
    review = preprocess_text(review)
    review_tfidf = vectorizer.transform([review])
    prediction = model.predict(review_tfidf)
    return prediction[0]

# Example usage
new_review = "This movie was absolutely fantastic! I loved it."
print("Sentiment:", predict_sentiment(new_review))



