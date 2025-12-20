
# =========================
# 0) Imports
# =========================
import pandas as pd
import json
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from scipy.sparse import csr_matrix

# =========================
# 1) Load data
# =========================
with open("data/train.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open("data/test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Train columns:", list(train_df.columns))
print("Test columns:", list(test_df.columns))

print("\nClass distribution (train):")
print(train_df["type"].value_counts())
print("\nClass proportions (train):")
print(train_df["type"].value_counts(normalize=True).round(3))

# =========================
# 2) Helpers
# =========================
TOKEN_RE = re.compile(r"[a-z0-9]+")

NEGATIONS = {
    "no", "not", "never", "none", "nobody", "nothing", "neither", "nowhere",
    "cannot", "can't", "dont", "don't", "doesnt", "doesn't", "didnt", "didn't",
    "isnt", "isn't", "arent", "aren't", "wasnt", "wasn't", "werent", "weren't",
    "won't", "wouldn't", "shouldn't", "couldn't", "mustn't"
}

def safe_str(x):
    return "" if x is None else str(x)

def normalize_text(s):
    s = safe_str(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens(s):
    return TOKEN_RE.findall(normalize_text(s))

def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0

def overlap_ratio(ans_set, other_set):
    # How much of answer is supported by other text (context/question)
    if not ans_set:
        return 0.0
    inter = len(ans_set & other_set)
    return inter / len(ans_set)

def negation_stats(text):
    t = tokens(text)
    neg_count = sum(1 for w in t if w in NEGATIONS)
    has_neg = 1 if neg_count > 0 else 0
    return neg_count, has_neg

# =========================
# 3) Create combined text for TF-IDF
# =========================
def combine_text(row):
    q = normalize_text(row.get("question", ""))
    c = normalize_text(row.get("context", ""))
    a = normalize_text(row.get("answer", ""))
    return f"QUESTION: {q}\nCONTEXT: {c}\nANSWER: {a}"

train_df["text"] = train_df.apply(combine_text, axis=1)
test_df["text"] = test_df.apply(combine_text, axis=1)

# =========================
# 4) Build linguistic feature matrix (overlap + negation + lengths)
# =========================
def build_linguistic_features(df: pd.DataFrame) -> csr_matrix:
    feats = []
    for _, row in df.iterrows():
        q = safe_str(row.get("question", ""))
        c = safe_str(row.get("context", ""))
        a = safe_str(row.get("answer", ""))

        q_t = set(tokens(q))
        c_t = set(tokens(c))
        a_t = set(tokens(a))

        # Overlap features
        jac_ac = jaccard(a_t, c_t)            # similarity between answer & context
        jac_aq = jaccard(a_t, q_t)            # similarity between answer & question
        ov_ac = overlap_ratio(a_t, c_t)       # % of answer tokens found in context
        ov_aq = overlap_ratio(a_t, q_t)       # % of answer tokens found in question

        # Negation features (mainly useful for contradiction)
        neg_a_count, neg_a_has = negation_stats(a)
        neg_c_count, neg_c_has = negation_stats(c)

        # Length features (sometimes irrelevant answers are very short/very long)
        a_len = len(normalize_text(a))
        c_len = len(normalize_text(c))
        q_len = len(normalize_text(q))

        feats.append([
            jac_ac, jac_aq, ov_ac, ov_aq,
            neg_a_count, neg_a_has, neg_c_count, neg_c_has,
            a_len, c_len, q_len
        ])

    X = np.array(feats, dtype=float)

    # Light scaling to keep magnitudes reasonable (optional but helpful)
    # We’ll scale lengths down to avoid dominating.
    X[:, 8] = X[:, 8] / 1000.0  # a_len
    X[:, 9] = X[:, 9] / 1000.0  # c_len
    X[:,10] = X[:,10] / 1000.0  # q_len

    return csr_matrix(X)

# Transformers that sklearn Pipeline can use
linguistic_transformer = FunctionTransformer(build_linguistic_features, validate=False)

# =========================
# 5) Train/validation split
# =========================
X = train_df[["question", "context", "answer", "text"]]  # pass df to feature builders
y = train_df["type"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 6) Model: TF-IDF + linguistic features (FeatureUnion)
# =========================

# TF-IDF should run on the "text" column only
def select_text_column(df):
    return df["text"].astype(str)

text_selector = FunctionTransformer(select_text_column, validate=False)

tfidf_branch = Pipeline([
    ("select_text", text_selector),
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    ))
])

# Linguistic features branch
ling_branch = Pipeline([
    ("ling", linguistic_transformer)
])

# Combine both feature spaces
features = FeatureUnion([
    ("tfidf_features", tfidf_branch),
    ("linguistic_features", ling_branch)
])

model = Pipeline([
    ("features", features),
    ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))
])

# Train + evaluate
model.fit(X_train, y_train)
val_preds = model.predict(X_val)

print("\n" + "="*70)
print("MODEL: TF-IDF + Overlap/Negation + Logistic Regression (balanced)")
print("="*70)
print(classification_report(y_val, val_preds, digits=3))
print("Confusion Matrix:\n", confusion_matrix(y_val, val_preds))
print("Macro F1:", round(f1_score(y_val, val_preds, average="macro"), 4))

# =========================
# 7) Train on full train and create submission
# =========================
model.fit(train_df[["question", "context", "answer", "text"]], train_df["type"])
test_preds = model.predict(test_df[["question", "context", "answer", "text"]])

# Use ID column from your test set
if "ID" in test_df.columns:
    submission = pd.DataFrame({"ID": test_df["ID"], "type": test_preds})
else:
    # fallback if ID isn't present
    submission = pd.DataFrame({"index": range(len(test_df)), "type": test_preds})

submission.to_csv("submission_linguistic.csv", index=False)
print("\nSaved submission_linguistic.csv ✅")
print(submission.head(10))
