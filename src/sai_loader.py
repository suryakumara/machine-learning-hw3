import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


def load_sai(path="../data/sai/train.csv"):
    """Read the Spooky Author Identification training file.

    Returns the list of sentences and the author labels encoded as 0, 1, 2.
    """
    df = pd.read_csv(path)
    texts = df["text"].tolist()
    encoder = LabelEncoder()
    labels = encoder.fit_transform(df["author"])  # EAP, HPL, MWS -> 0, 1, 2
    return texts, labels, encoder.classes_


def make_tfidf(max_features=20000, ngram_range=(1, 2)):
    """Create a TF-IDF vectorizer that turns each sentence into a vector.

    max_features: keep only the most frequent words / word pairs.
    ngram_range: (1, 2) uses single words and pairs of consecutive words.
    """
    return TfidfVectorizer(max_features=max_features,
                           ngram_range=ngram_range,
                           sublinear_tf=True)

if __name__ == "__main__":
    texts, labels, classes = load_sai()
    print("Number of sentences:", len(texts))
    print("Classes:", list(classes))
    print("Count per class:", pd.Series(labels).value_counts().sort_index().tolist())
    print("Example:", texts[0][:120], "->", classes[labels[0]])

    # Quick check only. In the experiments, fit the vectorizer inside each
    # cross-validation fold (e.g. with a Pipeline), not on all data at once.
    vec = make_tfidf()
    X = vec.fit_transform(texts)
    print("TF-IDF matrix:", X.shape, "| non-zero ratio:", round(X.nnz / (X.shape[0] * X.shape[1]), 5))