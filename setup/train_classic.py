import pandas as pd
import joblib
from .dataset import get_as_pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

df = get_as_pandas('train')

X = df['text']
emotion_cols = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
y = df[emotion_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_df=0.9,
        min_df=5,
        ngram_range=(1, 2),
        stop_words='english'
    )),
    ('clf', OneVsRestClassifier(
        LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced'
        )
    ))
])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, './model_classic.joblib')
print("Model saved to './model_classic.joblib'")