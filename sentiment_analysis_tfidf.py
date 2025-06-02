import nltk
import pandas as pd
import numpy as np
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import re

nlp = spacy.load("en_core_web_sm")

# Load the dataset
df_sa = pd.read_csv('P:/My Documents/Books & Research/Analytics Vidya Blackbelt program/NLP with Pytorch/training.1600000.processed.noemoticon.csv', encoding = 'latin1')
df_sa.info()
df_sa.columns
df_sa.dtypes
df_sa.shape
df_sa.head()
df_sa.describe()

print("\nMissing Values:\n", df_sa.isnull().sum())

def clean_text(text):
    """Remove HTML tags, special characters, and extra spaces from text."""
    text = str(text)
    text = re.sub(r'<.*?>', '', text)          # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)        # Keep only letters, numbers, and spaces
    text = re.sub(r'\s+', ' ', text).strip()   # Collapse multiple spaces
    return text
df_sa['clean_text'] = df_sa['text of the tweet '].apply(clean_text)

def preprocess_text(texts):
    # lemmatize the tokens and store them in a list
    processed_texts = []
    for doc in nlp.pipe(texts, n_process=-1):
        lemmatized_tokens = [token.lemma_.lower() for token in doc if
                             token.is_alpha and token.lemma_ not in nlp.Defaults.stop_words]

        # Join the lemmatized tokens into a string
        processed_text = " ".join(lemmatized_tokens)

        processed_texts.append(processed_text)

    return processed_texts

df_sa['processed_text'] = preprocess_text(df_sa['clean_text'])

# Prepare features and target
df_sa = df_sa.dropna(subset=['processed_text'])
X = df_sa['processed_text']
y = df_sa['polarity of tweet ']

# Convert polarity to binary (assuming 0=negative, 4=positive in original dataset)
y = y.replace(4, 1)  # Convert 4 (positive) to 1 for binary classification

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Scale the features (TF-IDF vectors are already normalized, but we can scale them further)
scaler = StandardScaler(with_mean=False)  
X_train_scaled = scaler.fit_transform(X_train_tfidf)
X_test_scaled = scaler.transform(X_test_tfidf)

# Train and evaluate models
def train_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{model_name} Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return model

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM" : LinearSVC(random_state=42, dual=False, max_iter=1000, tol=0.001,  C=0.5, verbose=1)
}

# Train and evaluate each model
trained_models = {}
for name, model in models.items():
    trained_model = train_evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, name)
    trained_models[name] = trained_model

# Function to predict sentiment of new text
def predict_sentiment(text, model, vectorizer, scaler):
    processed_text = preprocess_text([text])
    text_tfidf = vectorizer.transform(processed_text)
    text_scaled = scaler.transform(text_tfidf)
    prediction = model.predict(text_scaled)
    return "Positive" if prediction[0] == 1 else "Negative"


best_model = trained_models["Logistic Regression"]
sample_text1 = "I love this product! It's amazing."
sample_text2 = "I work 60 hours a week to be this poor."
sample_text3 = "Really, Sherlock? No! You are amazingly clever."
sample_text4 = ("Thank you for explaining that my eye cancer isn't going to make me deaf. "
                "I feel so fortunate that an intellectual giant like yourself would deign to operate on me.")

print(f"\nSample prediction for '{sample_text}':")
print(predict_sentiment(sample_text, best_model, tfidf_vectorizer, scaler))

print(f"\nSample prediction for '{sample_text2}':")
print(predict_sentiment(sample_text2, best_model, tfidf_vectorizer, scaler))

print(f"\nSample prediction for '{sample_text3}':")
print(predict_sentiment(sample_text3, best_model, tfidf_vectorizer, scaler))

print(f"\nSample prediction for '{sample_text4}':")
print(predict_sentiment(sample_text4, best_model, tfidf_vectorizer, scaler))

