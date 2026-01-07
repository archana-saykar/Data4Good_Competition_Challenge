# =========================
# FINAL SUBMISSION CODE (JSON output)
# - Trains on train.json
# - Predicts labels for test.json
# - Saves predictions to submission_final.json (required by submission form)
# =========================

# =========================
# 0) Imports
# =========================
import json
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC

# =========================
# 1) Load data
# =========================
with open("data/train.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open("data/test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# Basic checks (keep minimal)
assert {"question", "context", "answer", "type"}.issubset(set(train_df.columns)), "Train columns missing!"
assert {"ID", "question", "context", "answer"}.issubset(set(test_df.columns)), "Test columns missing!"
print("Loaded ✅ | Train:", train_df.shape, "| Test:", test_df.shape)

# =========================
# 2) Helpers (tokens, overlap, negation, numbers/years)
# =========================
TOKEN_RE = re.compile(r"[a-z0-9]+")
NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2}|21\d{2})\b")

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
    return set(NUM_RE.findall(normalize_text(text)))

def extract_years(text):
    return set(YEAR_RE.findall(normalize_text(text)))

# =========================
# 3) Linguistic feature builder
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
        num_jac = jaccard(a_nums, c_nums)
        num_ov = overlap_ratio(a_nums, c_nums)
        num_a_count = len(a_nums)
        num_c_count = len(c_nums)
        num_extra_in_answer = max(0, num_a_count - len(a_nums & c_nums))

        # Year features
        a_years = extract_years(a)
        c_years = extract_years(c)
        year_jac = jaccard(a_years, c_years)
        year_ov = overlap_ratio(a_years, c_years)
        year_a_count = len(a_years)
        year_c_count = len(c_years)
        year_extra_in_answer = max(0, year_a_count - len(a_years & c_years))

        feats.append([
            jac_ac, jac_aq, ov_ac, ov_aq,
            neg_a_count, neg_a_has, neg_c_count, neg_c_has,
            a_len, c_len, q_len,
            num_jac, num_ov, num_a_count, num_c_count, num_extra_in_answer,
            year_jac, year_ov, year_a_count, year_c_count, year_extra_in_answer
        ])

    X = np.array(feats, dtype=float)

    # Scale lengths
    X[:, 8]  /= 1000.0
    X[:, 9]  /= 1000.0
    X[:,10]  /= 1000.0

    # Scale count features slightly
    for idx in [13, 14, 15, 18, 19, 20]:
        X[:, idx] /= 10.0

    return csr_matrix(X)

linguistic_transformer = FunctionTransformer(build_linguistic_features, validate=False)

# =========================
# 4) Feature pipeline: Split TF-IDF Q/C/A + Answer char n-grams + Linguistic
# =========================
def select_question(df): return df["question"].astype(str)
def select_context(df):  return df["context"].astype(str)
def select_answer(df):   return df["answer"].astype(str)

q_selector = FunctionTransformer(select_question, validate=False)
c_selector = FunctionTransformer(select_context, validate=False)
a_selector = FunctionTransformer(select_answer, validate=False)

tfidf_q = Pipeline([
    ("select_q", q_selector),
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95))
])

tfidf_c = Pipeline([
    ("select_c", c_selector),
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95))
])

tfidf_a = Pipeline([
    ("select_a", a_selector),
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95))
])

tfidf_char_a = Pipeline([
    ("select_a", a_selector),
    ("tfidf", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3
    ))
])

ling_branch = Pipeline([
    ("ling", linguistic_transformer)
])

features = FeatureUnion([
    ("tfidf_question", tfidf_q),
    ("tfidf_context", tfidf_c),
    ("tfidf_answer", tfidf_a),
    ("tfidf_answer_char", tfidf_char_a),
    ("linguistic_features", ling_branch)
])

# =========================
# 5) Final Model: Linear SVM (best settings)
# =========================
final_model = Pipeline([
    ("features", features),
    ("clf", LinearSVC(class_weight="balanced", C=0.5))
])

# =========================
# 6) (Optional) Validation metrics for your report
# =========================
# Keep this ON while writing your report; you can comment it out for a super-clean final run.
X_all = train_df[["question", "context", "answer"]]
y_all = train_df["type"]

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

final_model.fit(X_train, y_train)
val_preds = final_model.predict(X_val)

print("\n" + "="*90)
print("FINAL MODEL: Split TF-IDF (Q/C/A) + Char n-grams + Linguistic + Linear SVM (C=0.5)")
print("="*90)
print(classification_report(y_val, val_preds, digits=3))
print("Confusion Matrix:\n", confusion_matrix(y_val, val_preds))
print("Macro F1:", round(f1_score(y_val, val_preds, average="macro"), 4))

# =========================
# 7) Train on full train + predict test
# =========================
final_model.fit(X_all, y_all)
test_preds = final_model.predict(test_df[["question", "context", "answer"]])

# =========================
# 8) Save submission as JSON (REQUIRED by submission instructions)
# Structure: list of {"ID": <int>, "type": <label>}
# =========================
submission = [
    {"ID": int(i), "type": str(t)}
    for i, t in zip(test_df["ID"].tolist(), test_preds.tolist())
]

# Sanity checks
assert len(submission) == len(test_df) == 2000, "Submission length mismatch!"
assert all(("ID" in r and "type" in r) for r in submission), "Missing keys in submission!"

with open("submission_final.json", "w", encoding="utf-8") as f:
    json.dump(submission, f, indent=2)

print("\nSaved submission_final.json ✅")
print("Preview:", submission[:5])
