#model_training.py

import pandas as pd
import joblib
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np
import nltk
from nltk.corpus import stopwords
import string

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(query):
    words = query.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def remove_punctuation(query):
    return query.translate(str.maketrans('', '', string.punctuation))

def preprocess_data(df):
    modified_df = pd.DataFrame({
        'Customer Query': df['Customer Query'],
        'Root Cause': df['Root Cause'],
        'Modified Query': [remove_stopwords(query) for query in df['Customer Query']]
    })
    modified_df['Modified Query'] = modified_df['Modified Query'].apply(remove_punctuation)
    modified_df.to_csv('modified_df.csv', index=False)
    return modified_df

def train_model(csv_file):
    df = pd.read_csv(csv_file)
    modified_df = preprocess_data(df)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(modified_df['Modified Query'])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(modified_df['Root Cause'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1, stratify=y)

    parameters = {'C': np.arange(0.1, 3.0, 0.2)}
    log_reg_classifier = LogisticRegression(max_iter=10000)
    grid_search_log_reg = GridSearchCV(log_reg_classifier, parameters, cv=5, scoring='f1_weighted', return_train_score=True, verbose=10, n_jobs=-1)
    grid_search_log_reg.fit(X_train, y_train)

    best_params_log_reg = grid_search_log_reg.best_params_
    best_C = best_params_log_reg['C']
    print('Best Parameters: ', best_params_log_reg)

    best_log_reg_model = LogisticRegression(C=best_C, max_iter=10000)
    best_log_reg_model.fit(X_train, y_train)

    # Save the trained model and preprocessors
    joblib.dump(best_log_reg_model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python model_training.py <csv_file_path>")
        sys.exit(1)

    csv_file_path = sys.argv[1]
    train_model(csv_file_path)