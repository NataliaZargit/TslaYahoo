# -*- coding: utf-8 -*-


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files


# upload all csv files which have been scraped from Yahoo Finance
uploaded = files.upload()

import glob

csv_files = glob.glob("tsla_conversations_filtered_*.csv")
print ("Files found:")
for f in csv_files:
    print(f)

# count the amount of files
print("Amount of files:")
print(len(csv_files))

# create dataframe with all comments
dfs = [pd.read_csv(f, sep=",", encoding="utf-8", on_bad_lines="skip", dtype={"comment_id": "Int64"}) for f in csv_files]

# checks
if not dfs:
    raise ValueError("The csv_files list is empty. There are no files to read.")

# concat
comments = pd.concat(dfs, axis=0, ignore_index=True)

print(comments.shape)
comments.head()

# data type in each column
print(comments.dtypes)

# # remove duplicates based on all columns
comments = comments.drop_duplicates()

# print the shape of the DataFrame (rows, columns)
print(comments.shape)

# Find duplicates based on created_at, created_time, author_id and message_text
duplicates = comments[comments.duplicated(subset=['created_at','created_time', 'author_id', 'message_text', 'sentiment'], keep=False)]

# Sort for easier inspection
duplicates = duplicates.sort_values(by=['created_at','created_time', 'author_id', 'message_text', 'sentiment'])

# Show first 20 rows
print(duplicates.head(20))

# remove duplicates based only on 'created_at','created_time', 'author_id', 'message_text', 'sentiment'
comments=comments.drop_duplicates(subset=['created_at','created_time', 'author_id', 'message_text', 'sentiment'])

# print the shape again to see the difference
print(comments.shape)

# find duplicates based on 'created_at','created_time', 'author_id', 'message_text'
duplicates = comments[comments.duplicated(subset=['created_at','created_time', 'author_id', 'message_text'], keep=False)]

# sort for easier inspection
duplicates = duplicates.sort_values(by=['created_at','created_time', 'author_id', 'message_text'])

# show first 20 rows
print(duplicates.head(20))

# check for missing values (NaN) in each column
comments.isna().sum()

# number of unique values ​​across all columns
comments.nunique()

# amount of unique values in 'sentiment' column
print(comments['sentiment'].value_counts())

# cross-check - amount of unique values in 'sentiment' column
count_brackets = (comments['sentiment'] == '[]').sum()

print("Number of '[]' in sentiment:", count_brackets)

count_bearish = (comments['sentiment'] == "['BEARISH']").sum()

print("Number of 'BEARISH' in sentiment:", count_bearish)

count_bullish = (comments['sentiment'] == "['BULLISH']").sum()

print("Number of 'BULLISH' in sentiment:", count_bullish)

count_neutral = (comments['sentiment'] == "['NEUTRAL']").sum()

print("Number of 'NEUTRAL' in sentiment:", count_neutral)

print (count_bearish+count_bullish+count_neutral+count_brackets)

# clean column "sentiment"
# replace empty string '[]' in column 'sentiment' with NaN
comments['sentiment'] = comments['sentiment'].replace(['[]'], np.nan)

# remove [], '', spaces in column "sentiment"
def extract_sentiment(tag):
    if pd.isna(tag):                                        # check if the value is NaN (missing)
        return np.nan
    tag = str(tag)                                          # convert the value to string
    tag = tag.replace("[", "").replace("]", "").replace("'", "").replace('"', "") # remove square brackets and quotes
    tag = tag.strip()                                       # remove leading and trailing spaces
    return tag if tag else np.nan                           # return cleaned tag, or NaN if empty

# apply the function to the 'sentiment' column and create a new column 'sentiment_clean'
comments["sentiment_clean"] = comments["sentiment"].apply(extract_sentiment)

rows_before = comments.shape[0]

print("Rows before:", rows_before)

# (cross-check) - amount of rows in sentiment column with NaN
print('\nAmount of rows in sentiment column with NaN:')
print(comments['sentiment'].isna().sum())

# delete rows where both 'message_text' and 'sentiment' are missing (NaN)
comments = comments.dropna(subset=["message_text", "sentiment"], how="all")

# print the shape of the DataFrame after deletion
rows_after = comments.shape[0]
print("\nRows after:", rows_after)

# Calculate how many rows were deleted
deleted_rows = rows_before - rows_after
print("\nDeleted rows:", deleted_rows)

print(comments['sentiment_clean'].value_counts())

# Convert timestamp to datetime ????
# comments['created_time'] = pd.to_datetime(comments['created_time'], errors='coerce')

comments.head()

# preparation to sentiment analysis - Binary Sentiment Label Mapping

# define a dictionary to map sentiment labels to binary values
label_map_bin = {"BULLISH": 1, "BEARISH": 0}

# create a boolean mask: True if sentiment_clean is in the dictionary keys
mask_labeled_bin = comments["sentiment_clean"].isin(label_map_bin.keys())

# filter the DataFrame to include only rows with BULLISH or BEARISH
labeled_bin = comments[mask_labeled_bin].copy()

# map sentiment labels to binary values and store in a new column 'label_bin'
labeled_bin["label_bin"] = labeled_bin["sentiment_clean"].map(label_map_bin).astype(int)

"""# Sentiment analysis"""

# Text prepararion - text cleaning

import re
import nltk

nltk.download("stopwords")
nltk.download('punkt_tab')
nltk.download("punkt")
nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))

# remove from stop_words 'up' and 'down'
stop_words.remove("up")
stop_words.remove("down")

lemmatizer = WordNetLemmatizer()

# create function
def clean_text(text):
    if pd.isna(text):   # if the value is NaN (missing), return empty string
        return ""
    text = text.lower()                         # lowercase
    text = re.sub(r"http\S+|www\S+", "", text ) # remove URLs
    text = re.sub(r"<.*?>", "", text)           # remove HTML tags
    text = re.sub(r"[^a-z0-9\s]", " ", text)    # remove specific symbols, keep spaces, keep only letters and numbers
    text = re.sub(r"\s+", " ", text)            # replace multiple spaces with a single space
    tokens = word_tokenize(text)                                 # tokenize
    tokens = [word for word in tokens if word not in stop_words] # remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]     # lemmatize
    return " ".join(tokens)                                      # join back

print("Stop words are:", stop_words)

#apply function
comments["clean_text"] = comments["message_text"].astype(str).apply(clean_text)

labeled_bin["clean_text"] = labeled_bin["message_text"].apply(clean_text)

print(comments.shape)

print(comments.head(20))

# ????
# remove short comments (less than 3 symbols) like "lol", "££", "$$", "??" and sentiment has not rate
# comments = comments[comments["clean_text"].str.len()>3] & (comments["sentiment"].notna())

# VADER
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# add some financial words into VADER library
sia.lexicon["bullish"] = 3.0
sia.lexicon["bearish"] = -3.0
sia.lexicon["long"] = 1.5
sia.lexicon["short"] = -1.5
sia.lexicon["call"] = 0.5
sia.lexicon["put"] = -0.5

# count compound score
labeled_bin["vader_score"] = labeled_bin["clean_text"].apply(
    lambda x: sia.polarity_scores(x)["compound"]
)

# comments["vader_sentiment"] = comments["clean_text"].apply(
#     lambda x: sia.polarity_scores(x)["compound"]
# )

# convert VADER-score to binary class
def vader_to_label(score):
    if score >= 0.05:
        return 1   # bullish / positive
    elif score <= -0.05:
        return 0   # bearish / negative
    else:
        return np.nan

labeled_bin["vader_pred"] = labeled_bin["vader_score"].apply(vader_to_label)
eval_vader = labeled_bin.dropna(subset=["vader_pred"])

# Comparison of VADER prediction with real labels
from sklearn.metrics import classification_report

print("VADER vs community BULLISH/BEARISH")
print(classification_report(eval_vader["label_bin"], eval_vader["vader_pred"].astype(int)))

"""VADER demonstrates substantial limitations when applied to financial discussion data.
Despite relatively high precision for the BEARISH class (0.77), it suffers from low recall (0.51) and extremely poor precision for the BULLISH class (0.35).
This indicates that VADER systematically misclassifies bullish investor sentiment and performs only marginally above chance overall (accuracy ≈ 0.55).
"""

# FinBERT

# Installation and initialization
!pip install transformers torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
model.eval()

# take a subsample of marked comments (to avoid running out of time)
sample = labeled_bin.sample(1000, random_state=42).copy()

# FinBERT batch prediction function
def finbert_predict_batch(texts, batch_size=32, max_length=128):
    all_scores = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size].tolist()
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length
        ).to(device)

        with torch.no_grad():
            outputs = model(**enc)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()  # shape: (batch, 3)
        all_scores.append(probs)

    return np.vstack(all_scores)

# run FinBERT on sample
scores = finbert_predict_batch(sample["clean_text"], batch_size=32)
# 0 = negative, 1 = neutral, 2 = positive
pred_ids = scores.argmax(axis=1)
sample["finbert_raw"] = pred_ids

# we will make a simple rule:
# 2 (positive) → 1 (bullish),
# 0 (negative) → 0 (bearish),
# 1 (neutral) → NaN (discard when comparing)

def finbert_to_label(x):
    if x == 2:
        return 1   # bullish
    elif x == 0:
        return 0   # bearish
    else:
        return np.nan

sample["finbert_pred"] = sample["finbert_raw"].apply(finbert_to_label)

eval_finbert = sample.dropna(subset=["finbert_pred"])

# FinBERT evaluation versus user-defined labels

print("FinBERT vs user BULLISH/BEARISH (sample)")
print(classification_report(eval_finbert["label_bin"], eval_finbert["finbert_pred"].astype(int)))

"""**Results**
Both VADER and FinBERT demonstrated poor alignment with community-labeled investor sentiment.
VADER achieved 55% accuracy, while FinBERT performed at 35%, indicating that neither lexicon-based nor financial news–based pretrained models capture the investment-oriented sentiment expressed by Yahoo Finance users.

FinBERT, despite being domain-specific for financial text, is trained on news polarity, not on investor sentiment.
Community labels such as BULLISH/BEARISH represent market expectations, not emotional tone, and therefore FinBERT systematically misclassifies bearish expectations as negative news sentiment.

**Logistic Regression/SVM?**
A supervised model trained directly on the community annotations might be an option to continue
"""

# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch


# tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
# model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# def get_finbert_sentiment(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True)
#     outputs = model(**inputs)
#     scores = torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy()[0]

#     # labels: 0 = negative, 1 = neutral, 2 = positive
#     sentiment_value = scores[2] - scores[0]  # positive - negative
#     return float(sentiment_value)

# comments["finbert_sentiment"] = comments["clean_text"].apply(get_finbert_sentiment)



# # check the lenght of the
# comments["len"] = comments["clean_comments"].str.len()
# comments["len"].describe()

# # Calculate mean and median
# mean_len = comments["len"].mean()
# median_len = comments["len"].median()

# # Create combined plot: histogram + boxplot
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8), gridspec_kw={'height_ratios':[4,1]})

# # Histogram on top
# ax1.hist(comments["len"], bins=50, color="skyblue", edgecolor="black")
# ax1.axvline(mean_len, color="red", linestyle="dashed", linewidth=2, label=f"Mean: {mean_len:.1f}")
# ax1.axvline(median_len, color="green", linestyle="dashed", linewidth=2, label=f"Median: {median_len:.1f}")
# ax1.set_title("Distribution of comment lengths")
# ax1.set_xlabel("Comment length (characters)")
# ax1.set_ylabel("Frequency")
# ax1.legend()

# # Boxplot below
# ax2.boxplot(comments["len"], vert=False, patch_artist=True,
#             boxprops=dict(facecolor="lightblue", color="black"),
#             medianprops=dict(color="red", linewidth=2),
#             whiskerprops=dict(color="black"),
#             capprops=dict(color="black"),
#             flierprops=dict(markerfacecolor="orange", marker="o", markersize=5))
# ax2.set_xlabel("Comment length (characters)")

# plt.tight_layout()
# plt.show()





