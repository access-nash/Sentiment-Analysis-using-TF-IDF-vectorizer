import nltk
import pandas as pd
import numpy as np
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import re

nlp = spacy.load("en_core_web_sm")

# Load the dataset
df_sa = pd.read_csv('P:/My Documents/Books & Research/Analytics Vidya Blackbelt program/NLP with Pytorch/training.1600000.processed.noemoticon.csv',encoding='latin1')


# Data preprocessing (unchanged)
def clean_text(text):
    """Remove HTML tags, special characters, and extra spaces from text."""
    text = str(text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


df_sa['clean_text'] = df_sa['text of the tweet '].apply(clean_text)


def preprocess_text(texts):
    processed_texts = []
    for doc in nlp.pipe(texts, n_process=-1):
        lemmatized_tokens = [token.lemma_.lower() for token in doc if
                             token.is_alpha and token.lemma_ not in nlp.Defaults.stop_words]
        processed_text = " ".join(lemmatized_tokens)
        processed_texts.append(processed_text)
    return processed_texts


df_sa['processed_text'] = preprocess_text(df_sa['clean_text'])


df_sa = df_sa.dropna(subset=['processed_text'])
X = df_sa['processed_text']
y = df_sa['polarity of tweet '].replace(4, 1)  # Convert to binary

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF Vectorization
print("\nPerforming TF-IDF vectorization...")
tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train_tfidf)
X_test_scaled = scaler.transform(X_test_tfidf)


# Enhanced train and evaluate function with hyperparameter tuning
def train_evaluate_model(model, param_grid, X_train, y_train, X_test, y_test, model_name, cv=3):
    print(f"\n=== Tuning {model_name} ===")

    # Create GridSearchCV object
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_

    # Evaluate
    y_pred = best_model.predict(X_test)
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
    print("\nTest Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return best_model


# Define models with parameter grids
models = {
    "Logistic Regression": {
        "model": LogisticRegression(random_state=42, max_iter=1000),
        "params": {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'class_weight': [None, 'balanced']
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            'n_estimators': [100, 200],
            'max_depth': [None, 50, 100],
            'min_samples_split': [2, 5],
            'class_weight': [None, 'balanced']
        }
    }
}

# Train and evaluate each model with hyperparameter tuning
trained_models = {}
for name, config in models.items():
    best_model = train_evaluate_model(
        config["model"],
        config["params"],
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        name
    )
    trained_models[name] = best_model


