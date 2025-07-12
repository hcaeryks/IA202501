from dataset import get_as_pandas
from preprocessing import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
import joblib

df = get_as_pandas('train')

X = df['text']
emotion_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise']
y = df[emotion_cols]

X_processed = preprocess_text(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_df=0.8,
        min_df=3,
        ngram_range=(1, 2),
        stop_words='english',
        max_features=10000
    )),
    ('clf', OneVsRestClassifier(
        LogisticRegression(
            solver='liblinear',
            max_iter=1000,
            class_weight='balanced',
            C=1.0
        )
    ))
])

parameters = {
    'tfidf__max_features': [1, 10000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__estimator__C': [0.1, 1, 10]
}

scorer = make_scorer(f1_score, average='micro')
grid_search = GridSearchCV(pipeline, parameters, scoring=scorer, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

final_model = grid_search.best_estimator_

try:
    joblib.dump(final_model, './model_classic.joblib', compress=1)
    print("Classic model saved to './model_classic.joblib'")
except Exception as e:
    print(f"Error saving model: {e}")