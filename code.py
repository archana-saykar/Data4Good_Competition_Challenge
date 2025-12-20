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
NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2}|21\d{2})\b")  # 1600-2199 (safe range)

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
    if not ans_set:
        return 0.0
    inter = len(ans_set & other_set)
    return inter / len(ans_set)

def negation_stats(text):
    t = tokens(text)
    neg_count = sum(1 for w in t if w in NEGATIONS)
    has_neg = 1 if neg_count > 0 else 0
    return neg_count, has_neg

def extract_numbers(text):
    # return set of stringified numbers like {"12", "3.5"}
    return set(NUM_RE.findall(normalize_text(text)))

def extract_years(text):
    return set(YEAR_RE.findall(normalize_text(text)))

# =========================
# 3) (Optional) combined text for reference/debug
# =========================
def combine_text(row):
    q = normalize_text(row.get("question", ""))
    c = normalize_text(row.get("context", ""))
    a = normalize_text(row.get("answer", ""))
    return f"QUESTION: {q}\nCONTEXT: {c}\nANSWER: {a}"

train_df["text"] = train_df.apply(combine_text, axis=1)
test_df["text"] = test_df.apply(combine_text, axis=1)

# =========================
# 4) Build linguistic feature matrix (overlap + negation + lengths + numbers/years)
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

        # Overlap features (text)
        jac_ac = jaccard(a_t, c_t)
        jac_aq = jaccard(a_t, q_t)
        ov_ac = overlap_ratio(a_t, c_t)
        ov_aq = overlap_ratio(a_t, q_t)

        # Negation features
        neg_a_count, neg_a_has = negation_stats(a)
        neg_c_count, neg_c_has = negation_stats(c)

        # Length features
        a_len = len(normalize_text(a))
        c_len = len(normalize_text(c))
        q_len = len(normalize_text(q))

        # Number features
        a_nums = extract_numbers(a)
        c_nums = extract_numbers(c)

        num_jac = jaccard(a_nums, c_nums)               # similarity of numbers
        num_ov = overlap_ratio(a_nums, c_nums)          # % of answer numbers seen in context
        num_a_count = len(a_nums)
        num_c_count = len(c_nums)
        num_extra_in_answer = max(0, num_a_count - len(a_nums & c_nums))  # answer numbers not in context

        # Year features
        a_years = extract_years(a)
        c_years = extract_years(c)

        year_jac = jaccard(a_years, c_years)
        year_ov = overlap_ratio(a_years, c_years)
        year_a_count = len(a_years)
        year_c_count = len(c_years)
        year_extra_in_answer = max(0, year_a_count - len(a_years & c_years))

        feats.append([
            # original
            jac_ac, jac_aq, ov_ac, ov_aq,
            neg_a_count, neg_a_has, neg_c_count, neg_c_has,
            a_len, c_len, q_len,

            # numbers
            num_jac, num_ov, num_a_count, num_c_count, num_extra_in_answer,

            # years
            year_jac, year_ov, year_a_count, year_c_count, year_extra_in_answer
        ])

    X = np.array(feats, dtype=float)

    # Scale lengths so they don't dominate
    X[:, 8]  = X[:, 8]  / 1000.0  # a_len
    X[:, 9]  = X[:, 9]  / 1000.0  # c_len
    X[:,10]  = X[:,10]  / 1000.0  # q_len

    # Optional mild scaling for counts to keep them comparable
    # (counts are usually small, but this prevents edge cases)
    for idx in [13,14,15,18,19,20]:  # num_a_count, num_c_count, num_extra, year_a_count, year_c_count, year_extra
        X[:, idx] = X[:, idx] / 10.0

    return csr_matrix(X)

linguistic_transformer = FunctionTransformer(build_linguistic_features, validate=False)

# =========================
# 5) Train/validation split
# =========================
X = train_df[["question", "context", "answer"]]
y = train_df["type"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 6) Model: Split TF-IDF (Q/C/A) + Linguistic features + numbers/years
# =========================
def select_question(df): return df["question"].astype(str)
def select_context(df):  return df["context"].astype(str)
def select_answer(df):   return df["answer"].astype(str)

q_selector = FunctionTransformer(select_question, validate=False)
c_selector = FunctionTransformer(select_context, validate=False)
a_selector = FunctionTransformer(select_answer, validate=False)

tfidf_q = Pipeline([
    ("select_q", q_selector),
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95))
])

tfidf_c = Pipeline([
    ("select_c", c_selector),
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95))
])

tfidf_a = Pipeline([
    ("select_a", a_selector),
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95))
])

ling_branch = Pipeline([
    ("ling", linguistic_transformer)
])

features_split = FeatureUnion([
    ("tfidf_question", tfidf_q),
    ("tfidf_context", tfidf_c),
    ("tfidf_answer", tfidf_a),
    ("linguistic_features", ling_branch)
])

model = Pipeline([
    ("features", features_split),
    ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))
])

# Train + evaluate
model.fit(X_train, y_train)
val_preds = model.predict(X_val)

print("\n" + "="*90)
print("MODEL: Split TF-IDF (Q/C/A) + Overlap/Negation + Numbers/Years + LR (balanced)")
print("="*90)
print(classification_report(y_val, val_preds, digits=3))
print("Confusion Matrix:\n", confusion_matrix(y_val, val_preds))
print("Macro F1:", round(f1_score(y_val, val_preds, average="macro"), 4))

# =========================
# 7) Train full + submission
# =========================
model.fit(train_df[["question", "context", "answer"]], train_df["type"])
test_preds = model.predict(test_df[["question", "context", "answer"]])

if "ID" in test_df.columns:
    submission = pd.DataFrame({"ID": test_df["ID"], "type": test_preds})
else:
    submission = pd.DataFrame({"index": range(len(test_df)), "type": test_preds})

submission.to_csv("submission_split_tfidf_linguistic_numbers.csv", index=False)
print("\nSaved submission_split_tfidf_linguistic_numbers.csv ✅")
print(submission.head(10))
