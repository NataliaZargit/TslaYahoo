# -*- coding: utf-8 -*-


# Import libraries
!pip -q install yfinance statsmodels imbalanced-learn
import os, re, glob, html
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import seaborn as sns
import matplotlib.pyplot as plt

from google.colab import files

import yfinance as yf

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 1) Upload CSV files


# upload all csv files which have been scraped from Yahoo Finance
uploaded = files.upload()

csv_files = glob.glob("tsla_conversations_filtered_*.csv")
print("Files found:", len(csv_files))
print("\n".join(csv_files[:5]), "..." if len(csv_files) > 5 else "")

dfs = []
for f in csv_files:
    df_part = pd.read_csv(
        f,
        sep=",",
        encoding="utf-8",
        on_bad_lines="skip",
        low_memory=False
    )
    dfs.append(df_part)

comments = pd.concat(dfs, axis=0, ignore_index=True)
print("Raw shape:", comments.shape)

comments.dtypes

# 2) Deduplication

# remove duplicates
comments = comments.drop_duplicates()

key_cols_1 = ['created_at','created_time','author_id','message_text','sentiment']
key_cols_1 = [c for c in key_cols_1 if c in comments.columns]
comments = comments.drop_duplicates(subset=key_cols_1)

key_cols_2 = ['created_at','created_time','author_id','message_text']
key_cols_2 = [c for c in key_cols_2 if c in comments.columns]
comments = comments.drop_duplicates(subset=key_cols_2)

print("After dedup:", comments.shape)

# 3) Clean sentiment column

# replace empty string '[]' in column 'sentiment' with NaN
comments['sentiment'] = comments['sentiment'].replace(['[]', ''], np.nan)

def extract_sentiment(tag):
    if pd.isna(tag):
        return np.nan
    tag = str(tag)
    tag = tag.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    tag = tag.strip()
    return tag if tag else np.nan

comments["sentiment_clean"] = comments["sentiment"].apply(extract_sentiment)

print(comments["sentiment_clean"].value_counts(dropna=False).head(10))

comments.head()

# 4) Clean message text (HTML + whitespace + tags)

TAG_RE = re.compile(r"<[^>]+>")

def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = html.unescape(s)         # &amp; -> &, &#39; -> '
    s = TAG_RE.sub(" ", s)       # remove HTML tags
    s = re.sub(r"\s+", " ", s).strip()
    return s

comments["text_clean"] = comments["message_text"].apply(clean_text)

# drop rows where cleaned text is empty
comments = comments[comments["text_clean"].str.len() > 0].copy()

print("After text cleaning:", comments.shape)

comments.head()

print("After text cleaning:", comments["sentiment_clean"].value_counts(dropna=False).head(10))

# 5) Split labelled vs unlabelled using "message_id"

# split the full dataset into:
# 1) labelled data  -> rows with user sentiment
# 2) unlabelled data -> EMPTY sentiment that we want to predict

# Sanity check: message_id must exist and be unique
assert "message_id" in comments.columns, "message_id column is missing!"

labelled = comments[comments["sentiment_clean"].notna()].copy()
unlabelled = comments[comments["sentiment_clean"].isna()].copy()

print("labelled rows:", labelled.shape[0])
print("Unlabelled (EMPTY) rows:", unlabelled.shape[0])
print("Unique message_id:", comments["message_id"].nunique())
print("Total rows:", comments.shape[0])

# Visualisation ---labelled vs Unlabelled Rows---

counts = {
    "Labelled": labelled.shape[0],
    "Unlabelled": unlabelled.shape[0]
}

plt.figure(figsize=(6, 5))

ax = sns.barplot(
    x=list(counts.keys()),
    y=list(counts.values())
)

plt.title("Labelled vs Unlabelled User Comment")
plt.xlabel("User Comment")
plt.ylabel("Number of Comments")

# Add values
for p in ax.patches:
    height = int(p.get_height())
    ax.annotate(
        f"{height}",
        (p.get_x() + p.get_width() / 2, height),
        ha="center",
        va="bottom",
        fontsize=12
    )

plt.tight_layout()
plt.show()

# 6) Train/Test split for EVALUATION for supervised learning (stratified)

# Text and labels for supervised learning
X = labelled["text_clean"]
y = labelled["sentiment_clean"]

# Stratified split (which is CRITICAL due to class imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y      # we divide into train and test while maintaining the proportions of classes (important for class imbalance).
)

# Visualisation ---Class Distribution---

print("labelled rows:", labelled.shape[0])
print("Train set:", X_train.shape[0])
print("Test set:", X_test.shape[0])

fig, ax = plt.subplots(figsize=(6, 5))

sns.countplot(x=y, ax=ax, order=y.value_counts().sort_values(ascending=False
).index)
ax.set_title("Class Distribution - Labelled")
ax.set_xlabel("Class")
ax.set_ylabel("Count")

# add values
for p in ax.patches:
    height = int(p.get_height())
    ax.annotate(
        f"{height}",
        (p.get_x() + p.get_width() / 2, height),
        ha="center",
        va="bottom",
        fontsize=11
    )

plt.tight_layout()
plt.show()


# Train/Test Class Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# TRAIN Class Distribution
sns.countplot(x=y_train, ax=axes[0], order=y_train.value_counts().sort_values(ascending=False
).index)
axes[0].set_title("Train Distribution")
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Count")

for p in axes[0].patches:
    height = int(p.get_height())
    axes[0].annotate(
        f"{height}",
        (p.get_x() + p.get_width() / 2, height),
        ha="center",
        va="bottom",
        fontsize=11
    )

# TEST Class Distribution
sns.countplot(x=y_test, ax=axes[1], order=y_test.value_counts().sort_values(ascending=False
).index)
axes[1].set_title("Test Distribution")
axes[1].set_xlabel("Class")
axes[1].set_ylabel("Count")

for p in axes[1].patches:
    height = int(p.get_height())
    axes[1].annotate(
        f"{height}",
        (p.get_x() + p.get_width() / 2, height),
        ha="center",
        va="bottom",
        fontsize=11
    )

plt.tight_layout()
plt.show()

# TF-IDF vectorization
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english",
    min_df=5,
    max_df=0.9,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 7) BASELINE MODEL - Logistic Regression (NO SMOTE, with class_weight="balanced")

# Logistic Regression
baseline_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    n_jobs=1,
    random_state=42
)

baseline_model.fit(X_train_vec, y_train)

# Predict class labels on the test set + Classification Report
y_pred_baseline = baseline_model.predict(X_test_vec)

print("""--- BASELINE (NO SMOTE, with class_weight="balanced") ---""")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred_baseline))

# Confusion Matrix
print("\n--- Confusion Matrix ---")
labels_order = baseline_model.classes_
cm_baseline = confusion_matrix(y_test, y_pred_baseline, labels=labels_order)
cm_baseline_df = pd.DataFrame(
    cm_baseline,
    index=[f"true_{c}" for c in labels_order],
    columns=[f"pred_{c}" for c in labels_order]
)
display(cm_baseline_df)

# BASELINE MODEL - Logistic Regression (SMOTE, without class_weight="balanced")

# SMOTE
smote_no_cw = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote_no_cw.fit_resample(X_train_vec, y_train)

# Logistic Regression
smote_no_cw_model = LogisticRegression(
    max_iter=2000,
    # class_weight="balanced",
    n_jobs=1,
    random_state=42
)

smote_no_cw_model.fit(X_train_resampled, y_train_resampled)

# Predict class labels on the test set
y_pred_smote_no_cw = smote_no_cw_model.predict(X_test_vec)

print("""--- BASELINE + SMOTE (without class_weight="balanced") ---""")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred_smote_no_cw))

print("\n--- Confusion Matrix ---")
labels_order = smote_no_cw_model.classes_
cm_smote_no_cw = confusion_matrix(y_test, y_pred_smote_no_cw, labels=labels_order)
cm_smote_no_cw_df = pd.DataFrame(
    cm_smote_no_cw,
    index=[f"true_{c}" for c in labels_order],
    columns=[f"pred_{c}" for c in labels_order]
)
display(cm_smote_no_cw_df)

# BASELINE MODEL - Logistic Regression (SMOTE, with class_weight="balanced")

# SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vec, y_train)

# Logistic Regression
smote_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    n_jobs=1,
    random_state=42
)

smote_model.fit(X_train_resampled, y_train_resampled)

# Predict class labels on the test set
y_pred_smote = smote_model.predict(X_test_vec)

print("""--- BASELINE + SMOTE (with class_weight="balanced") ---""")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred_smote))

print("\n--- Confusion Matrix ---")
labels_order = smote_model.classes_
cm_smote = confusion_matrix(y_test, y_pred_smote, labels=labels_order)
cm_smote_df = pd.DataFrame(
    cm_smote,
    index=[f"true_{c}" for c in labels_order],
    columns=[f"pred_{c}" for c in labels_order]
)
display(cm_smote_df)

# HYPERPARAMETER TUNING (GRID SEARCH (FULL PIPELINE) + SMOTE + with class_weight=”balanced”)

# End-to-end ML pipeline: TF-IDF → SMOTE → Logistic Regression
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", sublinear_tf=True)),
    ("smote", SMOTE(random_state=42)),
    ("clf", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        n_jobs=1,
        random_state=42
    ))
])

# Hyperparameter grid
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [3, 5, 10],
    "tfidf__max_df": [0.85, 0.90],
    "clf__C": [0.2, 1, 5]
}

# GridSearchCV
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit on TRAIN split only
grid.fit(X_train, y_train)

print("\n--- TUNING DONE ---")
print("Best params:", grid.best_params_)
print("Best CV macro-F1:", grid.best_score_)

# Evaluate tuned model on test split
tuned_model = grid.best_estimator_
y_pred_tuned = tuned_model.predict(X_test)

print("""\n--- TUNED MODEL (SMOTE, with class_weight=”balanced”) ---""")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred_tuned))

print("\n--- Confusion Matrix ---")
labels_order = tuned_model.named_steps["clf"].classes_
cm_tuned = confusion_matrix(y_test, y_pred_tuned, labels=labels_order)
cm_tuned_df = pd.DataFrame(
    cm_tuned,
    index=[f"true_{c}" for c in labels_order],
    columns=[f"pred_{c}" for c in labels_order]
)
display(cm_tuned_df)

# HYPERPARAMETER TUNING (GRID SEARCH (FULL PIPELINE) SMOTE + without class_weight=”balanced”)

# End-to-end ML pipeline: TF-IDF → SMOTE → Logistic Regression
pipe_no_cw = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", sublinear_tf=True)),
    ("smote", SMOTE(random_state=42)),
    ("clf", LogisticRegression(
        max_iter=3000,
        #class_weight="balanced",
        n_jobs=1,
        random_state=42
    ))
])

# Hyperparameter grid
param_grid_no_cw = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [3, 5, 10],
    "tfidf__max_df": [0.85, 0.90],
    "clf__C": [0.2, 1, 5]
}

# GridSearchCV
grid_no_cw = GridSearchCV(
    estimator=pipe_no_cw,
    param_grid=param_grid_no_cw,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit on TRAIN split only
grid_no_cw.fit(X_train, y_train)

print("\n--- TUNING DONE (SMOTE, without class_weight=”balanced”) ---")
print("Best params:", grid_no_cw.best_params_)
print("Best CV macro-F1:", grid_no_cw.best_score_)

# Evaluate tuned model on test split
tuned_model_no_cw = grid_no_cw.best_estimator_
y_pred_tuned_no_cw = tuned_model_no_cw.predict(X_test)

print("""\n--- TUNED MODEL (SMOTE, without class_weight=”balanced”) ---""")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred_tuned_no_cw))

print("\n--- Confusion Matrix ---")
labels_order = tuned_model_no_cw.named_steps["clf"].classes_
cm_tuned_no_cw = confusion_matrix(y_test, y_pred_tuned_no_cw, labels=labels_order)
cm_tuned_no_cw_df = pd.DataFrame(
    cm_tuned_no_cw,
    index=[f"true_{c}" for c in labels_order],
    columns=[f"pred_{c}" for c in labels_order]
)
display(cm_tuned_no_cw_df)

# HYPERPARAMETER TUNING (GRID SEARCH (FULL PIPELINE), no SMOTE, with class_weight="balanced)

# Pipeline WITHOUT SMOTE
pipe_no_smote = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", sublinear_tf=True)),
    ("clf", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",   # handle class imbalance via weights
        n_jobs=1,
        random_state=42
    ))
])

# Hyperparameter grid
param_grid_no_smote = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [3, 5, 10],
    "tfidf__max_df": [0.85, 0.90],
    "clf__C": [0.2, 1, 5]
}

# GridSearchCV
grid_no_smote = GridSearchCV(
    estimator=pipe_no_smote,
    param_grid=param_grid_no_smote,
    scoring="f1_macro",   # important for imbalanced classes
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit on TRAIN split only
grid_no_smote.fit(X_train, y_train)

print("\n--- TUNING DONE (NO SMOTE, class_weight='balanced') ---")
print("Best params:", grid_no_smote.best_params_)
print("Best CV macro-F1:", grid_no_smote.best_score_)

# Evaluate tuned model on TEST split
tuned_model_no_smote = grid_no_smote.best_estimator_
y_pred_tuned_no_smote = tuned_model_no_smote.predict(X_test)

print("\n--- TUNED MODEL (NO SMOTE, class_weight='balanced') ---")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred_tuned_no_smote))

print("\n--- Confusion Matrix ---")
labels_order = tuned_model_no_smote.named_steps["clf"].classes_
cm_tuned_no_smote = confusion_matrix(
    y_test,
    y_pred_tuned_no_smote,
    labels=labels_order
)

cm_tuned_no_smote_df = pd.DataFrame(
    cm_tuned_no_smote,
    index=[f"true_{c}" for c in labels_order],
    columns=[f"pred_{c}" for c in labels_order]
)

display(cm_tuned_no_smote_df)

# Visualisation---Confusion Matrix---
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_tuned_no_smote_df,
    annot=True,
    fmt="d",
    cmap="Blues",
    linewidths=0.5,
    linecolor="white"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")

plt.tight_layout()
plt.show()

# Save test set predictions for manual inspection
# Extract message_id for the test rows
test_evaluation = labelled.loc[X_test.index, ["message_id"]].copy()
# Add text, true labels, and model predictions
test_evaluation["text_clean"] = X_test
test_evaluation["true_label"] = y_test
test_evaluation["model_pred"] = y_pred_tuned_no_smote

# Save to csv
test_evaluation.to_csv("model_quality_check.csv", index=False)
files.download("model_quality_check.csv")

test_evaluation.head(10)

# 8) Predict sentiment labels for EMPTY rows using tuned_model_no_smote  + probabilities in EMPTY rows

Xu = unlabelled["text_clean"]

# Predict class probabilities and sentiment labels
proba = tuned_model_no_smote.predict_proba(Xu)     # probabilities for each class
pred = tuned_model_no_smote.predict(Xu)            # predicted sentiment labels

# Convert probabilities to DataFrame
proba_df = pd.DataFrame(
    proba,
    columns=[f"p_{c}" for c in tuned_model_no_smote.classes_]
)

# Build output dataframe for EMPTY rows
unlabelled_out = unlabelled[["message_id"]].copy()
unlabelled_out["pred_label"] = pred

unlabelled_out = pd.concat(
    [unlabelled_out.reset_index(drop=True),
     proba_df.reset_index(drop=True)],
    axis=1
)

# Create a numerical indicator for financial analysis (bullish vs bearish pressure) - Sentiment score
# Positive -> bullish pressure
# Negative -> bearish pressure

unlabelled_out["sentiment_score"] = (
    unlabelled_out["p_BULLISH"] - unlabelled_out["p_BEARISH"]
)

display(unlabelled_out.head())

# Visualisation --- Distribution of predicted labels ---
plt.figure(figsize=(7,6))

# Count predicted labels
label_counts = unlabelled_out["pred_label"].value_counts().reindex(tuned_model_no_smote.classes_)

sns.barplot(
    x=label_counts.index,
    y=label_counts.values
)

plt.title("Distribution of Predicted Sentiment Labels (Unlabelled Data)")
plt.xlabel("Predicted Label")
plt.ylabel("Count")

# Add values on top of each bar
for i, v in enumerate(label_counts.values):
    plt.text(i, v + max(label_counts.values)*0.01, str(v),
             ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.show()

print("Total Amount of Predicted Sentiment Labels:", len(unlabelled_out))

# Visualisation --- Combined Distribution: labelled vs Unlabelled ---

# Count labelled distribution
labelled_counts = labelled["sentiment_clean"].value_counts().reindex(tuned_model_no_smote.classes_)
labelled_percent = labelled_counts / labelled_counts.sum() * 100

# Count predicted distribution
pred_counts = unlabelled_out["pred_label"].value_counts().reindex(tuned_model_no_smote.classes_)
pred_percent = pred_counts / pred_counts.sum() * 100

# Build dataframe for plotting
dist_df = pd.DataFrame({
    "Label": tuned_model_no_smote.classes_,
    "labelled (%)": labelled_percent.values,
    "Predicted (%)": pred_percent.values
})

# Plot
plt.figure(figsize=(10,6))
bar_width = 0.35
x = range(len(dist_df))

plt.bar(x, dist_df["labelled (%)"], width=bar_width, label="labelled Data", alpha=0.8)
plt.bar([i + bar_width for i in x], dist_df["Predicted (%)"], width=bar_width, label="Predicted (Unlabelled)", alpha=0.8)

# Add values on top of bars
for i, v in enumerate(dist_df["labelled (%)"]):
    plt.text(i, v + 0.5, f"{v:.1f}%", ha='center')

for i, v in enumerate(dist_df["Predicted (%)"]):
    plt.text(i + bar_width, v + 0.5, f"{v:.1f}%", ha='center')

plt.xticks([i + bar_width/2 for i in x], dist_df["Label"])
plt.ylabel("Percentage (%)")
plt.title("Comparison of Sentiment Distribution: labelled vs Predicted (Unlabelled)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()

print("--- Total Amount of labelled Sentiment Data:", len(labelled))

print("--- Absolute Counts: labelled Data ---")
for label, count in labelled_counts.items():
    print(f"{label}: {count}")

print("\n--- Total Amount of Predicted Sentiment Labels:", len(unlabelled_out))

print("--- Absolute Counts: Predicted (Unlabelled Data) ---")
for label, count in pred_counts.items():
    print(f"{label}: {count}")

# Visualisation ---Distribution of Sentiment Score for Predicted labels---
plt.figure(figsize=(8, 5))

sns.histplot(
    unlabelled_out["sentiment_score"],
    bins=50,
    kde=True
)

plt.title("Distribution of Sentiment Score for Predicted Labels")
plt.xlabel("Sentiment Score (p_BULLISH - p_BEARISH)")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# 9) Merge predicted labels into full dataset (Final)

assert unlabelled_out["message_id"].is_unique
unlabelled_out = unlabelled_out.drop_duplicates(subset=["message_id"])
dup = comments["message_id"].duplicated().sum()
print("Duplicated message_id in comments:", dup)
comments = comments.drop_duplicates(subset=["message_id"])

# Merge predictions back using "message_id"
final = comments.merge(
    unlabelled_out,
    on="message_id",
    how="left"
)

# Fill sentiment:

# - use real user label when available
# - otherwise use model prediction
final["sentiment_filled"] = final["sentiment_clean"]
mask_empty = final["sentiment_filled"].isna()
final.loc[mask_empty, "sentiment_filled"] = final.loc[mask_empty, "pred_label"]

print("Final sentiment distribution:")
print(final["sentiment_filled"].value_counts(dropna=False))

# Visualisation ---"Final Sentiment Distribution" (Filled Labels - real and predicted labels)---

counts = final["sentiment_filled"].value_counts()

plt.figure(figsize=(7, 5))

ax = sns.barplot(
    x=counts.index,
    y=counts.values
)

plt.title("Final Sentiment Distribution (real and predicted labels)")
plt.xlabel("Sentiment Class")
plt.ylabel("Count")

# add values
for p in ax.patches:
    height = int(p.get_height())
    ax.annotate(
        f"{height}",
        (p.get_x() + p.get_width() / 2, height),
        ha="center",
        va="bottom",
        fontsize=12
    )

plt.tight_layout()
plt.show()

# 10) Save final dataset with filled sentiment and probabilities (this is master dataset)
final.to_csv("tsla_sentiment_filled_ml.csv", index=False)
final.to_parquet("tsla_sentiment_filled_ml.parquet", index=False)

files.download("tsla_sentiment_filled_ml.csv")

final.head(5)

# 11) BUILD DAILY SENTIMENT DATASETS

# Parse datetime (also convert from UTC to America/New_York)
final = final.copy()

final["created_dt"] = (pd.to_datetime(
    final["created_at"].astype(str) + " " + final["created_time"].astype(str),
    errors="coerce",
    utc=True
  ).dt.tz_convert("America/New_York")
)

# Drop rows where datetime couldn't be parsed
final = final[final["created_dt"].notna()].copy()
final["date"] = final["created_dt"].dt.date

# 12) Option A - DAILY AGGREGATION using only sentiment_filled (without avg_sentiment_score, we have share_BULLISH, share_BEARISH, share_NEUTRAL, share_spread = share_BULLISH - share_BEARISH)

daily_counts_A = (
    final.groupby(["date", "sentiment_filled"])
         .size()
         .unstack(fill_value=0)
)

for c in ["BEARISH", "BULLISH", "NEUTRAL"]:
    if c not in daily_counts_A.columns:
        daily_counts_A[c] = 0

daily_counts_A["n_comments"] = daily_counts_A.sum(axis=1)

daily_shares_A = daily_counts_A[["BEARISH","BULLISH","NEUTRAL"]].div(
    daily_counts_A["n_comments"], axis=0
).add_prefix("share_")

daily_A = pd.concat([daily_counts_A, daily_shares_A], axis=1).reset_index().sort_values("date")
daily_A["share_spread"] = daily_A["share_BULLISH"] - daily_A["share_BEARISH"]

daily_A.to_csv("tsla_daily_sentiment_option_A.csv", index=False)
files.download("tsla_daily_sentiment_option_A.csv")

daily_A.head(5)

# 13) Option B - add sentiment_score for labelled rows too, calculate daily avg sentiment score

# Predict probabilities for labelled rows (feature generation only, not replacing user labels)
X_labelled_all = labelled["text_clean"]

proba_labelled = tuned_model_no_smote.predict_proba(X_labelled_all)
proba_labelled_df = pd.DataFrame(proba_labelled, columns=[f"p_{c}" for c in tuned_model_no_smote.classes_])

labelled_out = labelled[["message_id"]].copy()
labelled_out = pd.concat([labelled_out.reset_index(drop=True), proba_labelled_df.reset_index(drop=True)], axis=1)
labelled_out["sentiment_score"] = labelled_out["p_BULLISH"] - labelled_out["p_BEARISH"]

# Merge labelled probabilities into final (only fill where p_* is missing)
final_B = final.merge(labelled_out, on="message_id", how="left", suffixes=("", "_labelled"))

for col in ["p_BEARISH", "p_BULLISH", "p_NEUTRAL", "sentiment_score"]:
    final_B[col] = final_B[col].combine_first(final_B[f"{col}_labelled"])

final_B = final_B.drop(columns=[c for c in final_B.columns if c.endswith("_labelled")])

final_B.head(5)

# Daily aggregation Option B: use class shares + avg/median sentiment_score

daily_counts_B = (
    final_B.groupby(["date", "sentiment_filled"])
          .size()
          .unstack(fill_value=0)
)

for c in ["BEARISH", "BULLISH", "NEUTRAL"]:
    if c not in daily_counts_B.columns:
        daily_counts_B[c] = 0

daily_counts_B["n_comments"] = daily_counts_B.sum(axis=1)

daily_shares_B = daily_counts_B[["BEARISH","BULLISH","NEUTRAL"]].div(
    daily_counts_B["n_comments"], axis=0
).add_prefix("share_")

daily_scores_B = final_B.groupby("date").agg(
    avg_sentiment_score=("sentiment_score", "mean"),
    median_sentiment_score=("sentiment_score", "median")
)

daily_B = (
    pd.concat([daily_counts_B, daily_shares_B, daily_scores_B], axis=1)
      .reset_index()
      .sort_values("date")
)

daily_B["share_spread"] = daily_B["share_BULLISH"] - daily_B["share_BEARISH"]

daily_B.to_csv("tsla_daily_sentiment_option_B.csv", index=False)
files.download("tsla_daily_sentiment_option_B.csv")

daily_B.head(5)

# TSLA PRICES (Compute Close-to-Close Returns)

# Download TSLA daily prices
tsla_prices = yf.download(
    "TSLA",
    period="max",
    interval="1d",
    auto_adjust=False,
    progress=False
).reset_index()

# If yfinance returns MultiIndex columns, flatten FIRST
if isinstance(tsla_prices.columns, pd.MultiIndex):
    tsla_prices.columns = [c[0] if isinstance(c, tuple) else c for c in tsla_prices.columns]

# Validate required columns
if "Date" not in tsla_prices.columns:
    raise ValueError(f"Unexpected yfinance output columns: {tsla_prices.columns}")

if "Close" not in tsla_prices.columns:
    raise ValueError(f"'Close' column not found. Columns: {tsla_prices.columns}")

# Create merge key
tsla_prices["date"] = pd.to_datetime(tsla_prices["Date"], errors="coerce").dt.date
tsla_prices = tsla_prices[tsla_prices["date"].notna()].copy()

# Compute close-to-close returns
tsla_prices = tsla_prices.sort_values("date").copy()
tsla_prices["close_to_close_return"] = tsla_prices["Close"].pct_change()

# Merge daily sentiment with TSLA returns
merged_A = daily_A.merge(
    tsla_prices[["date", "Close", "close_to_close_return"]],
    on="date",
    how="inner"
).sort_values("date")

merged_B = daily_B.merge(
    tsla_prices[["date", "Close", "close_to_close_return"]],
    on="date",
    how="inner"
).sort_values("date")

print("Merged_A shape:", merged_A.shape)
print("Merged_B shape:", merged_B.shape)
print("Date range A:", merged_A["date"].min(), "->", merged_A["date"].max())
print("Date range B:", merged_B["date"].min(), "->", merged_B["date"].max())

tsla_prices.head()

print(merged_A.columns)
print(merged_B.columns)

merged_A.head()

merged_B.head()

# Correlations (same-day)

# Option A uses share_spread = share_BULLISH - share_BEARISH
corr_A_pearson = merged_A[["share_spread", "close_to_close_return"]].corr(method="pearson").iloc[0, 1]
corr_A_spearman = merged_A[["share_spread", "close_to_close_return"]].corr(method="spearman").iloc[0, 1]

# Option B uses avg_sentiment_score (mean of pBULLISH - pBEARISH)
corr_B_pearson = merged_B[["avg_sentiment_score", "close_to_close_return"]].corr(method="pearson").iloc[0, 1]
corr_B_spearman = merged_B[["avg_sentiment_score", "close_to_close_return"]].corr(method="spearman").iloc[0, 1]

print("\n--- Same-day correlations ---")
print("Option A (share_spread vs return)  Pearson:", corr_A_pearson, " Spearman:", corr_A_spearman)
print("Option B (avg_score vs return)  Pearson:", corr_B_pearson, " Spearman:", corr_B_spearman)

# Lagged correlations: sentiment(t) vs return(t+1..t+5)

def lagged_corr(df, sentiment_col, return_col="close_to_close_return", max_lag=5):
    out = []
    for lag in range(0, max_lag + 1):
        # return at t+lag
        shifted_return = df[return_col].shift(-lag)
        tmp = pd.DataFrame({"sent": df[sentiment_col], "ret": shifted_return}).dropna()
        out.append({
            "lag_days": lag,
            "pearson": tmp.corr(method="pearson").iloc[0, 1] if len(tmp) > 2 else np.nan,
            "spearman": tmp.corr(method="spearman").iloc[0, 1] if len(tmp) > 2 else np.nan,
            "n": len(tmp)
        })
    return pd.DataFrame(out)

lags_A = lagged_corr(merged_A, "share_spread", max_lag=5)
lags_B = lagged_corr(merged_B, "avg_sentiment_score", max_lag=5)

print("\n--- Lagged correlations: sentiment(t) vs return(t+lag) ---")
print("\nOption A: share spread vs close-to-close return")
display(lags_A)
print("\nOption B: avg sentiment score vs close-to-close return")
display(lags_B)

# Save merged outputs
merged_A.to_csv("tsla_daily_sentiment_with_returns_A.csv", index=False)
merged_B.to_csv("tsla_daily_sentiment_with_returns_B.csv", index=False)

files.download("tsla_daily_sentiment_with_returns_A.csv")
files.download("tsla_daily_sentiment_with_returns_B.csv")

# Visualisation ---Rolling Correlations (30-day and 90-day)---

# Option A
merged_A = merged_A.sort_values("date").copy()
merged_A["roll_corr_30"] = (
    merged_A["share_spread"]
    .rolling(window=30)
    .corr(merged_A["close_to_close_return"])
)
merged_A["roll_corr_90"] = (
    merged_A["share_spread"]
    .rolling(window=90)
    .corr(merged_A["close_to_close_return"])
)

plt.figure(figsize=(12,4))
plt.plot(merged_A["date"], merged_A["roll_corr_30"], label="30-day rolling corr")
plt.plot(merged_A["date"], merged_A["roll_corr_90"], label="90-day rolling corr")
plt.axhline(0, color="black", linewidth=1)
plt.title("Rolling Correlation (Option A): Sentiment Spread vs Returns")
plt.legend()
plt.tight_layout()
plt.show()

# Option B
merged_B = merged_B.sort_values("date").copy()
merged_B["roll_corr_30"] = (
    merged_B["avg_sentiment_score"]
    .rolling(window=30)
    .corr(merged_B["close_to_close_return"])
)
merged_B["roll_corr_90"] = (
    merged_B["avg_sentiment_score"]
    .rolling(window=90)
    .corr(merged_B["close_to_close_return"])
)

plt.figure(figsize=(12,4))
plt.plot(merged_B["date"], merged_B["roll_corr_30"], label="30-day rolling corr")
plt.plot(merged_B["date"], merged_B["roll_corr_90"], label="90-day rolling corr")
plt.axhline(0, color="black", linewidth=1)
plt.title("Rolling Correlation (Option B): Avg Sentiment Score vs Returns")
plt.legend()
plt.tight_layout()
plt.show()

# Visualisation --- Sentiment vs daily returns (Scatter + regression line)
def plot_sent_vs_return_regplot(df, xcol, ycol="close_to_close_return", title=""):
    d = df[[xcol, ycol]].copy()
    d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
    d = d.dropna()

    pearson = d.corr(method="pearson").iloc[0, 1]
    spearman = d.corr(method="spearman").iloc[0, 1]

    plt.figure(figsize=(7, 5))
    ax = sns.regplot(
        data=d,
        x=xcol,
        y=ycol,
        scatter_kws={"alpha": 0.4}
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.title(title)

    # Write correlations on plot
    ax.text(
        0.02, 0.95,
        f"Pearson = {pearson:.3f}\nSpearman = {spearman:.3f}\nn = {len(d)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.tight_layout()
    plt.show()

# Option A: share_spread vs daily returns (regression line)
plot_sent_vs_return_regplot(
    merged_A,
    xcol="share_spread",
    title="Option A: share_spread vs daily returns (regression line)"
)

#O ption B: avg_sentiment_score vs daily returns (regression line)
plot_sent_vs_return_regplot(
    merged_B,
    xcol="avg_sentiment_score",
    title="Option B: avg_sentiment_score vs daily returns (regression line)"
)

# Correlation with |returns| (Absolute Returns = Volatility Proxy)

# Visualisation ---Option A: Share Spread vs Absolute Returns (Volatility Proxy)---
# Visualisation ---Option B: Avg Sentiment Score vs Absolute Returns (Volatility Proxy)---

# 1) Create absolute returns (volatility proxy)
merged_A = merged_A.copy()
merged_B = merged_B.copy()

merged_A["abs_ret"] = merged_A["close_to_close_return"].abs()
merged_B["abs_ret"] = merged_B["close_to_close_return"].abs()

# 2) Create function: compute Pearson + Spearman + n
def corr_abs(df, feature, target="abs_ret"):
    tmp = df[[feature, target]].dropna()
    return {
        "feature": feature,
        "pearson": tmp.corr(method="pearson").iloc[0, 1],
        "spearman": tmp.corr(method="spearman").iloc[0, 1],
        "n": len(tmp)
    }

# 3) Build a results table (Option A + Option B)
rows = []
rows.append({**corr_abs(merged_A, "share_spread"), "option": "A", "description": "Share spread (Bullish - Bearish)"})
rows.append({**corr_abs(merged_A, "n_comments"),   "option": "A", "description": "Activity (number of comments)"})
rows.append({**corr_abs(merged_B, "avg_sentiment_score"), "option": "B", "description": "Avg sentiment score (pBullish - pBearish)"})
rows.append({**corr_abs(merged_B, "n_comments"),          "option": "B", "description": "Activity (number of comments)"})

corr_table_abs = pd.DataFrame(rows)[["option", "feature", "description", "pearson", "spearman", "n"]]
corr_table_abs["pearson"] = corr_table_abs["pearson"].round(4)
corr_table_abs["spearman"] = corr_table_abs["spearman"].round(4)

print("\n=== Correlation with |returns| (Absolute Returns as Volatility Proxy) ===")
display(corr_table_abs)

# 4) Visualization
# Option A: share_spread vs abs_ret
plt.figure(figsize=(7, 4))
sns.regplot(
    data=merged_A,
    x="share_spread",
    y="abs_ret",
    scatter_kws={"alpha": 0.35},
    line_kws={"color": "red"}
)
plt.title("Option A: Share Spread vs Absolute Returns (Volatility Proxy)")
plt.xlabel("Share Spread = share_BULLISH - share_BEARISH")
plt.ylabel("|Return| (Absolute daily return)")
plt.tight_layout()
plt.show()

# Option B: avg_sentiment_score vs abs_ret
plt.figure(figsize=(7, 4))
sns.regplot(
    data=merged_B,
    x="avg_sentiment_score",
    y="abs_ret",
    scatter_kws={"alpha": 0.35},
    line_kws={"color": "red"}
)
plt.title("Option B: Avg Sentiment Score vs Absolute Returns (Volatility Proxy)")
plt.xlabel("Avg Sentiment Score = p(BULLISH) - p(BEARISH)")
plt.ylabel("|Return| (Absolute daily return)")
plt.tight_layout()
plt.show()

# OLS Regression with Controls (Option A and Option B)

# Dependent variable: close_to_close_return
# Controls: lag_return (t-1), log_n_comments
# Robust SE: HC3

def run_ols_lagvol(df, sent_col, label):
    reg = df.copy().sort_values("date")

    # Ensure numeric columns (prevents dtype=object errors)
    for col in [sent_col, "close_to_close_return", "n_comments"]:
        reg[col] = pd.to_numeric(reg[col], errors="coerce")

    # Controls
    reg["lag_return"] = reg["close_to_close_return"].shift(1)                 # return(t-1)
    reg["log_n_comments"] = np.log1p(reg["n_comments"])                       # log(1 + comments)

    # Drop NA rows needed for regression
    reg = reg.dropna(subset=[sent_col, "close_to_close_return", "lag_return", "log_n_comments"]).copy()

    # Define X and y
    X = reg[[sent_col, "lag_return", "log_n_comments"]]
    X = sm.add_constant(X)
    y = reg["close_to_close_return"]

    # Fit OLS with robust SE
    model = sm.OLS(y, X).fit(cov_type="HC3")

    print("\n")
    print(f"OLS with Controls (lagged vol): {label}")
    print(f"Sentiment variable: {sent_col}")
    print("n =", len(reg))

    print(model.summary())

    return model, reg

# Run Option A (share_spread)
mA_lagvol, regA_lagvol = run_ols_lagvol(merged_A, "share_spread", "Option A (share_spread)")

# Run Option B (avg_sentiment_score)
mB_lagvol, regB_lagvol = run_ols_lagvol(merged_B, "avg_sentiment_score", "Option B (avg_sentiment_score)")

# ARIMAX / SARIMAX
# Robustness check: ARIMAX
# Option B: avg_sentiment_score

arimax_df = merged_B.copy().sort_values("date")

# Ensure numeric
arimax_df["avg_sentiment_score"] = pd.to_numeric(arimax_df["avg_sentiment_score"], errors="coerce")
arimax_df["close_to_close_return"] = pd.to_numeric(arimax_df["close_to_close_return"], errors="coerce")

# Drop NA
arimax_df = arimax_df.dropna(subset=["avg_sentiment_score", "close_to_close_return"])

# Endogenous variable (returns)
y = arimax_df["close_to_close_return"]

# Exogenous variable (sentiment)
X = arimax_df[["avg_sentiment_score"]]

# Fit simple ARIMAX(1,0,0)
arimax_model = SARIMAX(
    endog=y,
    exog=X,
    order=(1, 0, 0),              # p=AR(1), d=no differencing, q=no MA
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

print("\n--- ARIMAX (Option B) summary ---")
print(arimax_model.summary())

# Visualisation ---In-sample fitted vs actual (illustrative)---
arimax_df["arimax_fitted"] = arimax_model.fittedvalues

plt.figure(figsize=(12, 5))
plt.plot(arimax_df["date"], arimax_df["close_to_close_return"], label="Actual returns", color="black", linewidth=1)
plt.plot(arimax_df["date"], arimax_df["arimax_fitted"], label="ARIMAX fitted", color="red", alpha=0.7)

plt.title("ARIMAX in-sample fit (Sentiment as exogenous variable)")
plt.xlabel("Date")
plt.ylabel("Daily return")
plt.legend()
plt.tight_layout()
plt.show()

"""The ARIMAX fitted values are shown for illustration only. The model is not used for forecasting but to assess whether sentiment remains statistically significant after accounting for time-series dependence"""

# INTRADAY (HOURLY) ANALYSIS

# 1) Build intraday comment-level dataset with a single timestamp column in NY time
# We will use final_B because it contains sentiment_score for all rows (labelled + predicted).

intraday = final_B.copy()

# created_dt is already NY time in the pipeline:
assert "created_dt" in intraday.columns, "created_dt is missing."


# Round timestamps down to the hour for aggregation
intraday["hour"] = intraday["created_dt"].dt.floor("h")


# 2) Hourly aggregation

# Option A: shares/spread from sentiment_filled
hourly_counts = (
    intraday.groupby(["hour", "sentiment_filled"])
            .size()
            .unstack(fill_value=0)
)

for c in ["BEARISH", "BULLISH", "NEUTRAL"]:
    if c not in hourly_counts.columns:
        hourly_counts[c] = 0

hourly_counts["n_comments"] = hourly_counts.sum(axis=1)

hourly_shares = hourly_counts[["BEARISH","BULLISH","NEUTRAL"]].div(hourly_counts["n_comments"], axis=0)
hourly_shares = hourly_shares.add_prefix("share_")

hourly_A = pd.concat([hourly_counts, hourly_shares], axis=1).reset_index()
hourly_A["share_spread"] = hourly_A["share_BULLISH"] - hourly_A["share_BEARISH"]


# Option B: probability-based score (mean/median sentiment_score)
hourly_scores = intraday.groupby("hour").agg(
    avg_sentiment_score=("sentiment_score", "mean"),
    median_sentiment_score=("sentiment_score", "median")
).reset_index()


# Merge A + B hourly features together
hourly = hourly_A.merge(hourly_scores, on="hour", how="left").sort_values("hour")

print("Hourly sentiment rows (Option A+B) :", hourly.shape)
display(hourly.head())


# 3) Download TSLA intraday prices (60m)
# 60m usually covers up to ~730 days (enough for 1-year window)
tsla_1h = yf.download(
    "TSLA",
    period="730d",
    interval="60m",
    auto_adjust=False,
    progress=False
)


# Flatten MultiIndex if needed (check if column is tuple)
if isinstance(tsla_1h.columns, pd.MultiIndex):
    tsla_1h.columns = [c[0] if isinstance(c, tuple) else c for c in tsla_1h.columns]

tsla_1h = tsla_1h.reset_index()


# As yfinance sometimes uses 'Datetime' as the column after reset_index for intraday (check if column name is "Datetime" or "Date", otherwise - error)
dt_col = None
for candidate in ["Datetime", "Date"]:
    if candidate in tsla_1h.columns:
        dt_col = candidate
        break
if dt_col is None:
    raise ValueError(f"Unexpected intraday yfinance columns: {tsla_1h.columns}")


# Convert TSLA timestamps to NY time and floor to hour to match hourly sentiment bins
tsla_1h["tsla_dt"] = pd.to_datetime(tsla_1h[dt_col], errors="coerce", utc=True).dt.tz_convert("America/New_York")
tsla_1h = tsla_1h[tsla_1h["tsla_dt"].notna()].copy()
tsla_1h["hour"] = tsla_1h["tsla_dt"].dt.floor("h")


# Compute intraday close-to-close returns per hour
if "Close" not in tsla_1h.columns:
    raise ValueError(f"'Close' column not found in intraday TSLA data. Columns: {tsla_1h.columns}")

tsla_1h = tsla_1h.sort_values("hour").copy()
tsla_1h["ret_1h"] = tsla_1h["Close"].pct_change()                # Calculate hourly returns


# Keep only what we need for merge
tsla_1h_small = tsla_1h[["hour", "Close", "ret_1h"]].drop_duplicates(subset=["hour"])

print("\nTSLA 60m rows:", tsla_1h_small.shape)
display(tsla_1h_small.head())


# 4) Merge hourly sentiment with hourly TSLA returns
merged_1h = hourly.merge(tsla_1h_small, on="hour", how="inner").sort_values("hour")

print("\nMerged intraday shape:", merged_1h.shape)
print("Date range:", merged_1h["hour"].min(), "->", merged_1h["hour"].max())
display(merged_1h.head())


# 5) Intraday correlations (same-hour)
def corr_table(df, cols, target="ret_1h"):
    out = []
    for c in cols:
        tmp = df[[c, target]].dropna()
        out.append({
            "feature": c,
            "pearson": tmp.corr(method="pearson").iloc[0,1],
            "spearman": tmp.corr(method="spearman").iloc[0,1],
            "n": len(tmp)
        })
    return pd.DataFrame(out).sort_values("spearman", ascending=False)

print("\n=== Intraday (1h) correlations with ret_1h ===")
display(corr_table(
    merged_1h,
    cols=["share_spread", "avg_sentiment_score", "median_sentiment_score", "n_comments"],
    target="ret_1h"
))

# 6) Lagged intraday correlations: sentiment(t) vs return(t+lag_hours)
def lagged_corr_hours(df, sentiment_col, return_col="ret_1h", max_lag=6):
    out = []
    for lag in range(0, max_lag + 1):
        shifted_return = df[return_col].shift(-lag)  # return at t+lag
        tmp = pd.DataFrame({"sent": df[sentiment_col], "ret": shifted_return}).dropna()
        out.append({
            "lag_hours": lag,
            "pearson": tmp.corr(method="pearson").iloc[0,1] if len(tmp) > 2 else np.nan,
            "spearman": tmp.corr(method="spearman").iloc[0,1] if len(tmp) > 2 else np.nan,
            "n": len(tmp)
        })
    return pd.DataFrame(out)

print("\n=== Lagged intraday correlations: share_spread(t) vs ret_1h(t+lag) ===")
display(lagged_corr_hours(merged_1h, "share_spread", max_lag=6))

print("\n=== Lagged intraday correlations: avg_sentiment_score(t) vs ret_1h(t+lag) ===")
display(lagged_corr_hours(merged_1h, "avg_sentiment_score", max_lag=6))

# 7) Save merged intraday dataset
merged_1h.to_csv("tsla_intraday_1h_sentiment_with_returns.csv", index=False)
files.download("tsla_intraday_1h_sentiment_with_returns.csv")

# DAY SENTIMENT vs SAME-DAY CLOSE & EVENING SENTIMENT vs NEXT-DAY OPEN

df = final_B.copy()

# Stock market trading hours (NYSE/Nasdaq)
market_open  = pd.to_datetime("09:30").time()
market_close = pd.to_datetime("16:00").time()

df["date"] = df["created_dt"].dt.date
df["time"] = df["created_dt"].dt.time

# Daytime sentiment (expectations for the market close)
df["is_day"] = df["time"].between(market_open, pd.to_datetime("15:59").time())

day_sent = (
    df[df["is_day"]]
    .groupby("date")
    .agg(
        day_spread=("sentiment_score", "mean"),
        day_share_spread=("sentiment_filled",
                          lambda x: (x=="BULLISH").mean() - (x=="BEARISH").mean()),
        n_day=("sentiment_filled", "count")
    )
    .reset_index()
)

# Evening sentiment (expectations for the next day’s market open)
df["is_evening"] = (df["time"] > market_close) | (df["time"] < market_open)

evening_sent = (
    df[df["is_evening"]]
    .groupby("date")
    .agg(
        evening_spread=("sentiment_score", "mean"),
        evening_share_spread=("sentiment_filled",
                              lambda x: (x=="BULLISH").mean() - (x=="BEARISH").mean()),
        n_evening=("sentiment_filled", "count")
    )
    .reset_index()
)

# Download TSLA daily prices

tsla = yf.download("TSLA", period="2y", interval="1d", auto_adjust=False)

# If columns are a MultiIndex, flatten them to a single level
if isinstance(tsla.columns, pd.MultiIndex):
    tsla.columns = ['_'.join([str(c) for c in col if c]) for col in tsla.columns]

tsla = tsla.reset_index()
tsla["date"] = tsla["Date"].dt.date


# Merge day sentiment with close TSLA prices
merged_day = day_sent.merge(tsla, on="date", how="inner")

merged_day["ret_open_to_close"] = (
    merged_day["Close_TSLA"] / merged_day["Open_TSLA"] - 1
)


# Merge evening sentiment with open TSLA prices
merged_evening = evening_sent.merge(tsla, on="date", how="inner")

merged_evening["next_open"] = merged_evening["Open_TSLA"].shift(-1)

merged_evening["ret_close_to_next_open"] = (
    merged_evening["next_open"] / merged_evening["Close_TSLA"] - 1
)


# Correlations (Pearson)
print("\n=== Day sentiment vs same-day close ===")
print(merged_day[["day_spread", "day_share_spread", "ret_open_to_close"]].corr())

print("\n=== Evening sentiment vs next-day open ===")
print(merged_evening[["evening_spread", "evening_share_spread", "ret_close_to_next_open"]].corr())

# Visualisation --- Scatter: Day sentiment vs same-day return ---
plt.figure(figsize=(7,5))
sns.regplot(
    data=merged_day,
    x="day_share_spread",
    y="ret_open_to_close",
    scatter_kws={"alpha":0.5, "color":"#4C72B0"},
    line_kws={"color":"#C44E52"}
)
plt.title(f"Day Sentiment vs Same-Day Return\nPearson = {merged_day[['day_share_spread','ret_open_to_close']].corr().iloc[0,1]:.3f}")
plt.xlabel("Day Share Spread (Bullish - Bearish)")
plt.ylabel("Return: Open → Close")
plt.tight_layout()
plt.show()

# Visualisation --- Scatter: Evening sentiment vs next-day open ---
plt.figure(figsize=(7,5))
sns.regplot(
    data=merged_evening,
    x="evening_share_spread",
    y="ret_close_to_next_open",
    scatter_kws={"alpha":0.5, "color":"#4C72B0"},
    line_kws={"color":"#C44E52"}
)
plt.title(f"Evening Sentiment vs Next-Day Open Return\nPearson = {merged_evening[['evening_share_spread','ret_close_to_next_open']].corr().iloc[0,1]:.3f}")
plt.xlabel("Evening Share Spread (Bullish - Bearish)")
plt.ylabel("Return: Close → Next Open")
plt.tight_layout()
plt.show()

# Visualisation --- Rolling correlation for day sentiment ---

merged_day = merged_day.sort_values("date")
merged_day["roll_corr_day"] = (
    merged_day["day_share_spread"]
    .rolling(window=20)
    .corr(merged_day["ret_open_to_close"])
)

plt.figure(figsize=(10,4))
plt.plot(merged_day["date"], merged_day["roll_corr_day"], color="#4C72B0")
plt.axhline(0, color="black", linewidth=1)
plt.title("Rolling 20-Day Correlation: Day Sentiment vs Same-Day Return")
plt.xlabel("Date")
plt.ylabel("Correlation")
plt.tight_layout()
plt.show()

# --- Rolling correlation for evening sentiment ---

merged_evening = merged_evening.sort_values("date")
merged_evening["roll_corr_evening"] = (
    merged_evening["evening_share_spread"]
    .rolling(window=20)
    .corr(merged_evening["ret_close_to_next_open"])
)

plt.figure(figsize=(10,4))
plt.plot(merged_evening["date"], merged_evening["roll_corr_evening"], color="#4C72B0")
plt.axhline(0, color="black", linewidth=1)
plt.title("Rolling 20-Day Correlation: Evening Sentiment vs Next-Day Open Return")
plt.xlabel("Date")
plt.ylabel("Correlation")
plt.tight_layout()
plt.show()

# Visualisation ---Daily Average Sentiment Score vs TSLA Closing Price---

plt.figure(figsize=(16,6))
plt.plot(merged_B["date"], merged_B["avg_sentiment_score"], label="Avg Sentiment Score", color="blue")
plt.ylabel("Avg Sentiment Score", color="blue")
plt.xlabel("Date")

plt.twinx()
plt.plot(merged_B["date"], merged_B["Close"], label="TSLA Close Price", color="red", alpha=0.6)
plt.ylabel("TSLA Close Price", color="red")

plt.title("Daily Average Sentiment Score vs TSLA Closing Price")
plt.show()

# Visualisation ---Daily Sentiment Spread vs TSLA Closing Price---

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(16,6))
plt.plot(merged_A["date"], merged_A["share_spread"], label="Sentiment Spread", color="blue")
plt.ylabel("Sentiment Spread", color="blue")
plt.xlabel("Date")

plt.twinx()
plt.plot(merged_A["date"], merged_A["Close"], label="TSLA Close Price", color="red", alpha=0.6)
plt.ylabel("TSLA Close Price", color="red")

plt.title("Daily Sentiment Spread vs TSLA Closing Price")
plt.show()

# Visualisation ---Local Extrema of Sentiment Spread (5‑day window) vs TSLA Close Price---

from scipy.signal import argrelextrema

# 1) Extract series
dates = merged_A["date"]
spread = merged_A["share_spread"].values

# 2) Find local maxima and minima in a 5‑day window
window = 5

local_max_idx = argrelextrema(spread, np.greater, order=window)[0]
local_min_idx = argrelextrema(spread, np.less, order=window)[0]

# 3) Plot
plt.figure(figsize=(16,6))

# Plot TSLA Close on secondary axis
ax1 = plt.gca()
ax2 = ax1.twinx()

# Plot TSLA Close
ax2.plot(dates, merged_A["Close"], color="blue", alpha=0.6, label="TSLA Close Price")
ax2.set_ylabel("TSLA Close Price", color="blue")

# Plot extrema points
ax1.scatter(dates.iloc[local_max_idx], spread[local_max_idx],
            color="green", s=60, label="Local Max (Spread)")
ax1.scatter(dates.iloc[local_min_idx], spread[local_min_idx],
            color="red", s=60, label="Local Min (Spread)")

# Connect extrema with lines
ax1.plot(dates.iloc[local_max_idx], spread[local_max_idx],
         color="green", linewidth=1.5, label="Max Line")
ax1.plot(dates.iloc[local_min_idx], spread[local_min_idx],
         color="red", linewidth=1.5, label="Min Line")

ax1.set_ylabel("Sentiment Spread (Extrema Only)", color="blue")
ax1.set_xlabel("Date")

plt.title("Local Extrema of Sentiment Spread (5‑day window) vs TSLA Close Price")

# Legends
ax1.legend(loc="lower left")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()

#Visualisation ---Local Extrema of Sentiment Spread vs Shifted TSLA Close Price---

from scipy.signal import argrelextrema

# 1) Extract series
dates = merged_A["date"]
spread = merged_A["share_spread"].values

#  2) Find local maxima and minima in a 5‑day window
window = 5
local_max_idx = argrelextrema(spread, np.greater, order=window)[0]
local_min_idx = argrelextrema(spread, np.less, order=window)[0]

#  3) Shift TSLA price left
shift_days = 14
shifted_close = merged_A["Close"].shift(-shift_days)

#  4) Plot
plt.figure(figsize=(16,6))

ax1 = plt.gca()
ax2 = ax1.twinx()

# Plot shifted TSLA Close
ax2.plot(dates, shifted_close, color="blue", alpha=0.6, label=f"TSLA Close (shifted {shift_days} days left)")
ax2.set_ylabel("TSLA Close Price", color="blue")

# Plot extrema points
ax1.scatter(dates.iloc[local_max_idx], spread[local_max_idx],
            color="green", s=60, label="Local Max (Spread)")
ax1.scatter(dates.iloc[local_min_idx], spread[local_min_idx],
            color="red", s=60, label="Local Min (Spread)")

# Connect extrema with lines
ax1.plot(dates.iloc[local_max_idx], spread[local_max_idx],
         color="green", linewidth=1.5)
ax1.plot(dates.iloc[local_min_idx], spread[local_min_idx],
         color="red", linewidth=1.5)

ax1.set_ylabel("Sentiment Spread (Extrema Only)", color="blue")
ax1.set_xlabel("Date")

plt.title("Local Extrema of Sentiment Spread vs Shifted TSLA Close Price")

ax1.legend(loc="lower left")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()

# BEST RESULTS: Best GREEN correlation & Best RED correlation

results = []

spread = merged_A["share_spread"].values
close = merged_A["Close"]

for window in range(1, 31):  # window 1..30

    #  Find extrema for this window
    local_max_idx = argrelextrema(spread, np.greater, order=window)[0]
    local_min_idx = argrelextrema(spread, np.less, order=window)[0]

    #  Create empty series
    max_series = pd.Series(np.nan, index=merged_A.index)
    min_series = pd.Series(np.nan, index=merged_A.index)

    # Fill extrema
    max_series.iloc[local_max_idx] = spread[local_max_idx]
    min_series.iloc[local_min_idx] = spread[local_min_idx]

    #  Interpolate
    max_interp = max_series.interpolate(method="linear")
    min_interp = min_series.interpolate(method="linear")

    #  Iterate over shift_days
    for shift_days in range(0, 21):

        shifted_close = close.shift(-shift_days)

        df = pd.DataFrame({
            "max_interp": max_interp,
            "min_interp": min_interp,
            "shifted_close": shifted_close
        }).dropna()

        if len(df) > 5:
            corr_max = df["max_interp"].corr(df["shifted_close"])
            corr_min = df["min_interp"].corr(df["shifted_close"])
        else:
            corr_max = np.nan
            corr_min = np.nan

        results.append((window, shift_days, corr_max, corr_min))

print("window | shift | corr_max | corr_min")
for window, shift_days, corr_max, corr_min in results:
    print(f"{window:6d} | {shift_days:5d} | {corr_max:8.4f} | {corr_min:8.4f}")

#  Find best correlations
best_max = max(results, key=lambda x: x[2] if not np.isnan(x[2]) else -999)
best_min = min(results, key=lambda x: x[3] if not np.isnan(x[3]) else 999)

print("\n=== BEST RESULTS ===")
print(f"Best GREEN maxima correlation:")
print(f"  window={best_max[0]}, shift={best_max[1]}, corr_max={best_max[2]:.4f}")

print(f"\nBest RED minima correlation:")
print(f"  window={best_min[0]}, shift={best_min[1]}, corr_min={best_min[3]:.4f}")

# Visualisation ---Avg_sentiment_score vs close-to-close returns---
plt.figure(figsize=(16,6))
plt.scatter(merged_B["avg_sentiment_score"], merged_B["close_to_close_return"], alpha=0.4)
plt.xlabel("Avg Sentiment Score")
plt.ylabel("Close-to-Close Return")
plt.title("Sentiment vs Daily Returns")
plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)
plt.show()

# Visualisation ---Share Spread vs Daily Returns---
plt.figure(figsize=(16,6))
sns.regplot(
    x=merged_A["share_spread"],
    y=merged_A["close_to_close_return"],
    scatter_kws={"alpha":0.4}
)
plt.xlabel("Share Spread (Bullish - Bearish)")
plt.ylabel("Daily Return")
plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)
plt.title("Share Spread vs Daily Returns")
plt.show()

# Visualisation ---90-Day Rolling Correlation: Sentiment vs Returns---
merged_A["rolling_corr"] = (
    merged_A["share_spread"]
    .rolling(90)
    .corr(merged_A["close_to_close_return"])
)

plt.figure(figsize=(16,6))
plt.plot(merged_A["date"], merged_A["rolling_corr"], color="purple")
plt.title("90-Day Rolling Correlation: Sentiment vs Returns")
plt.ylabel("Correlation")
plt.xlabel("Date")
plt.axhline(0, color="black", linewidth=1)
plt.show()

# Visualisation ---30-Day Rolling Correlation: Sentiment vs Returns---
merged_B["rolling_corr"] = (
    merged_B["avg_sentiment_score"]
    .rolling(30)
    .corr(merged_B["close_to_close_return"])
)

plt.figure(figsize=(16,6))
plt.plot(merged_B["date"], merged_B["rolling_corr"], color="purple")
plt.title("30-Day Rolling Correlation: Sentiment vs Returns")
plt.ylabel("Correlation")
plt.xlabel("Date")
plt.axhline(0, color="black", linewidth=1)
plt.show()
