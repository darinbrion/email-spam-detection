"""Spam/Ham Email Classifier

A cleaned, GitHub-ready notebook combining text exploration, feature engineering, and spam classification modeling.
"""


# # Spam/Ham Email Classifier  A cleaned, GitHub-ready notebook combining text exploration, feature engineering, and spam classification modeling.

# ## Exploratory analysis
# Run this cell to suppress all FutureWarnings.
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# More readable exceptions.

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set(style = "whitegrid",
        color_codes = True,
        font_scale = 1.5)

import zipfile

# Loading training and test datasets
with zipfile.ZipFile('spam_ham_data.zip') as item:
    with item.open("train.csv") as f:
        original_training_data = pd.read_csv(f)
    with item.open("test.csv") as f:
        test = pd.read_csv(f)

# Convert the emails to lowercase as the first step of text processing.
original_training_data['email'] = original_training_data['email'].str.lower()
test['email'] = test['email'].str.lower()

original_training_data.head()

print('Before imputation:')
print(original_training_data.isnull().sum())
original_training_data = original_training_data.fillna('')
print('------------')
print('After imputation:')
print(original_training_data.isnull().sum())

example_ham = original_training_data.loc[original_training_data['spam'] == 0, 'email'].iloc[0]
example_spam = original_training_data.loc[original_training_data['spam'] == 1, 'email'].iloc[37]
print("Ham Email:")
print(example_ham)
print("-------------------------------------------------")
print("Spam Email:")
print(example_spam)

# This creates a 90/10 train-validation split on our labeled data.
from sklearn.model_selection import train_test_split

train, val = train_test_split(original_training_data, test_size=0.1, random_state=42)

def words_in_texts(words, texts):
    """
    Args:
        words (list): Words to find.
        texts (Series): Strings to search in.

    Returns:
        A 2D NumPy array of 0s and 1s with shape (n, d) where
        n is the number of texts, and d is the number of words.
    """
    indicator_array = np.array([texts.str.contains(w).astype(int) for w in words]).T
    return indicator_array

# Run this cell to see what your function outputs. Compare the results to the example provided above.
words_in_texts(['hello', 'bye', 'world'], pd.Series(['hello', 'hello worldhello']))

from IPython.display import display, Markdown
df = pd.DataFrame({
    'word_1': [1, 0, 1, 0],
    'word_2': [0, 1, 0, 1],
    'type': ['spam', 'ham', 'ham', 'ham']
})
display(Markdown("> Our original `DataFrame` has a `type` column and some columns corresponding to words. You can think of each row as a sentence, and the value of 1 or 0 indicates whether the word appears in this sentence (1 if present, 0 if not)."))
display(df);
display(Markdown("> `melt` will turn columns into entries in a variable column. Notice how `word_1` and `word_2` become entries in `variable`; their values are stored in the `value` column."))
display(df.melt("type"))

words = ['click', 'free', 'offer', 'money', 'credit', 'investment']

df = words_in_texts(words, train["email"])
df = pd.DataFrame(df, columns = words)
df['type'] = ['spam' if value == 1.0 else "ham" for value in train['spam']]
melted_df = df.melt("type")

train = train.reset_index(drop=True) # We must do this in order to preserve the ordering of emails to labels for words_in_texts.

plt.figure(figsize=(8,6))

sns.set(font_scale=1.2)
sns.set_style("whitegrid")

sns.barplot(data = melted_df, x = 'variable', y = "value" , hue = 'type', errorbar = None)

plt.title('Frequency of Words in Spam/Ham Emails')
plt.xlabel('Words')
plt.xticks(rotation=35)
plt.ylabel('Proportion of Emails')
plt.ylim(0, 1)

plt.legend(
    title=None,
    loc="upper right"
)

plt.tight_layout()
plt.show()

some_words = ['drug', 'bank', 'prescription', 'memo', 'private']

X_train = words_in_texts(some_words, train["email"])
Y_train = np.array(train["spam"])

X_train[:5], Y_train[:5]

from sklearn.linear_model import LogisticRegression

my_model = LogisticRegression()
my_model.fit(X_train, Y_train)

training_accuracy = my_model.score(X_train, Y_train)
print("Training Accuracy: ", training_accuracy)

assert np.allclose(my_model.coef_, np.array([[ 0.38812285 , 1.41514963 , 2.04579703 , -0.53358517 , 0.91970759]]))

zero_predictor_fp = 0
zero_predictor_fn = sum(train["spam"] == 1.0)
zero_predictor_fp, zero_predictor_fn

tp = 0
tn = sum(train["spam"] == 0.0)
fp = 0
fn = sum(train["spam"] == 1.0)

zero_predictor_acc = (tp + tn) / (tp + fp + tn + fn)
zero_predictor_recall = 0
zero_predictor_acc, zero_predictor_recall

Y_train_hat = my_model.predict(X_train)

TP = sum((Y_train_hat == 1.0) & (Y_train == 1.0))
TN = sum((Y_train_hat == 0.0) & (Y_train == 0.0))
FP = sum((Y_train_hat == 1.0) & (Y_train == 0.0))
FN = sum((Y_train_hat == 0.0) & (Y_train == 1.0))

logistic_predictor_precision = TP / (TP + FP)
logistic_predictor_recall = TP / (TP + FN)
logistic_predictor_fpr = FP / (FP + TN)

print(f"{TP=}, {TN=}, {FP=}, {FN=}")
print(f"{logistic_predictor_precision=:.2f}, {logistic_predictor_recall=:.2f}, {logistic_predictor_fpr=:.2f}")

q6e = True
q6e


# ## Model development
# Run this cell to suppress all FutureWarnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set(style = "whitegrid",
        color_codes = True,
        font_scale = 1.5)

from datetime import datetime
from IPython.display import display, HTML

import zipfile
with zipfile.ZipFile('spam_ham_data.zip') as item:
    with item.open("train.csv") as f:
        original_training_data = pd.read_csv(f)
    with item.open("test.csv") as f:
        test = pd.read_csv(f)

# Convert the emails to lowercase as the first step of text processing.
original_training_data['email'] = original_training_data['email'].str.lower()
test['email'] = test['email'].str.lower()

original_training_data.head()

# Fill any missing or NAN values.
print('Before imputation:')
print(original_training_data.isnull().sum())
original_training_data = original_training_data.fillna('')
print('------------')
print('After imputation:')
print(original_training_data.isnull().sum())

# This creates a 90/10 train-validation split on our labeled data.
from sklearn.model_selection import train_test_split
train, val = train_test_split(original_training_data, test_size = 0.1, random_state = 42)

# We must do this in order to preserve the ordering of emails to labels for words_in_texts.
train = train.reset_index(drop = True)

from projB2_utils import *

words_in_texts(['hello', 'bye', 'world'], pd.Series(['hello', 'hello worldhello']))

some_words = ['drug', 'bank', 'prescription', 'memo', 'private']

X_train = words_in_texts(some_words, train['email'])
Y_train = np.array(train['spam'])

X_train[:5], Y_train[:5]

from sklearn.linear_model import LogisticRegression

simple_model = LogisticRegression()
simple_model.fit(X_train, Y_train)

training_accuracy = simple_model.score(X_train, Y_train)
print("Training Accuracy: ", training_accuracy)

train_c = train.copy()

train_c['num_excl'] = train_c['email'].str.count('!')
train_c['num_chars'] = train_c['email'].str.len()

train_c['log_excl'] = np.log1p(train_c['num_excl'])
train_c['log_chars'] = np.log1p(train_c['num_chars'])

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=train_c.sample(2000, random_state=0),  # sample for clarity
    x='log_excl',
    y='log_chars',
    hue='spam',
    alpha=0.5,
    palette='coolwarm'
)

plt.xlabel("log(1 + # of Exclamation Marks)")
plt.ylabel("log(1 + Email Length)")
plt.title("Relationship Between Exclamation Usage and Email Length")
plt.show()

# import libraries
# You may use any of these to create your features.
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import re
from collections import Counter

# 1. Base numeric and structural features
train['num_chars'] = train['email'].str.len()
train['num_excl'] = train['email'].str.count('!')
train['num_q'] = train['email'].str.count('\?')
train['num_digits'] = train['email'].str.count('\d')

# Avoiding division by zero for densities
chars_nozero = train['num_chars'].replace(0, np.nan)

train['excl_density'] = (train['num_excl'] / chars_nozero).fillna(0)
train['q_density'] = (train['num_q'] / chars_nozero).fillna(0)
train['digit_density'] = (train['num_digits'] / chars_nozero).fillna(0)

# HTML indicator
train['has_html'] = train['email'].str.contains('<html', case=False).astype(int)

# Subject line information
train['subject_len'] = train['subject'].str.len()
train['subject_excl'] = train['subject'].str.count('!')
train['subject_question'] = train['subject'].str.count('\?')

# URLs
train['num_urls'] = train['email'].str.count(r'http[s]?://')
train['url_density'] = (train['num_urls'] / chars_nozero).fillna(0)

# all caps words
train['num_allcaps_words'] = train['email'].str.findall(r'\b[A-Z]{2,}\b').str.len()
train['allcaps_density'] = (train['num_allcaps_words'] / chars_nozero).fillna(0)

# Short subject flag
train['short_subject'] = (train['subject'].str.len() < 20).astype(int)

# Broad keyword hit
keywords = ['free', 'win', 'money', 'offer', 'click', 'credit', 'buy now', 'sex']
pattern = '|'.join(keywords)
train['keyword_hit'] = train['email'].str.contains(pattern, case=False).astype(int)

# 2.  spammy words from B1
manual_spam_words = [
    'offer', 'money', 'credit', 'investment',
    'click', 'please', 'html', 'business', 'body'
]

for word in manual_spam_words:
    train[f'has_{word}'] = train['email'].str.contains(rf"\b{word}\b", case=False).astype(int)

# 3. Build feature matrix X and defining target Y
base_feature_cols = [
    'num_chars', 'num_excl', 'num_q', 'num_digits',
    'excl_density', 'q_density', 'digit_density',
    'has_html', 'keyword_hit',
    'subject_len', 'subject_excl', 'subject_question',
    'num_urls', 'url_density',
    'num_allcaps_words', 'allcaps_density',
    'short_subject'
]

manual_word_feature_cols = [f"has_{w}" for w in manual_spam_words]

# Combining and removing duplicates while preserving order
feature_cols = list(dict.fromkeys(base_feature_cols + manual_word_feature_cols))

X = train[feature_cols].astype(float).copy()
y = train['spam']

# 4. Manualscaling
X_scaled = X.copy()

for col in X_scaled.columns:
    col_min = X_scaled[col].min()
    col_max = X_scaled[col].max()
    if X_scaled[col].nunique() <= 1:
        X_scaled[col] = 0.0
    else:
        X_scaled[col] = (X_scaled[col] - col_min) / (col_max - col_min)

# 5. Train logistic regression with L1 and GridSearchCV
log_reg = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    max_iter=5000
)

param_grid = {
    'C': [0.01, 0.1, 1, 10]
}

grid = GridSearchCV(log_reg, param_grid, cv=5)
grid.fit(X_scaled, y)

best_model = grid.best_estimator_

train_predictions = best_model.predict(X_scaled)

# Print your training accuracy.
training_accuracy = np.mean(train_predictions == train["spam"])
training_accuracy

# See a 10 fold CV for the simple 5 word model from the intro. Feel free to edit this cell.
compute_CV_error(X_train, train["spam"], folds=10)

# numeric and structural features
test['num_chars'] = test['email'].str.len()
test['num_excl'] = test['email'].str.count('!')
test['num_q'] = test['email'].str.count('\?')
test['num_digits'] = test['email'].str.count('\d')

# Avoiding division by zero for densities
test_chars_nozero = test['num_chars'].replace(0, np.nan)

test['excl_density'] = (test['num_excl'] / test_chars_nozero).fillna(0)
test['q_density'] = (test['num_q'] / test_chars_nozero).fillna(0)
test['digit_density'] = (test['num_digits'] / test_chars_nozero).fillna(0)

# HTML indicator
test['has_html'] = test['email'].str.contains('<html', case=False).astype(int)

# Subject line information
test['subject_len'] = test['subject'].str.len()
test['subject_excl'] = test['subject'].str.count('!')
test['subject_question'] = test['subject'].str.count('\?')

# URLs
test['num_urls'] = test['email'].str.count(r'http[s]?://')
test['url_density'] = (test['num_urls'] / test_chars_nozero).fillna(0)

# all caps words
test['num_allcaps_words'] = test['email'].str.findall(r'\b[A-Z]{2,}\b').str.len()
test['allcaps_density'] = (test['num_allcaps_words'] / test_chars_nozero).fillna(0)

# Short subject flag
test['short_subject'] = (test['subject'].str.len() < 20).astype(int)

# Broad keyword hit
keywords = ['free', 'win', 'money', 'offer', 'click', 'credit', 'buy now', 'sex']
pattern = '|'.join(keywords)
test['keyword_hit'] = test['email'].str.contains(pattern, case=False).astype(int)

# spam-indicative words
manual_spam_words = [
    'offer', 'money', 'credit', 'investment',
    'click', 'please', 'html', 'business', 'body'
]

for word in manual_spam_words:
    test[f'has_{word}'] = test['email'].str.contains(rf"\b{word}\b", case=False).astype(int)

# 2. X_test with the same features as in training
X_test = test[feature_cols].astype(float).copy()

# 3. scaling test features
X_train_for_scaling = train[feature_cols].astype(float).copy()
train_mins = X_train_for_scaling.min()
train_maxs = X_train_for_scaling.max()

X_test_scaled = X_test.copy()
for col in X_test_scaled.columns:
    col_min = train_mins[col]
    col_max = train_maxs[col]
    if col_max == col_min:
        X_test_scaled[col] = 0.0
    else:
        X_test_scaled[col] = (X_test_scaled[col] - col_min) / (col_max - col_min)

X_test_scaled = X_test_scaled.fillna(0)

# 4. getting predictions
test_predictions = best_model.predict(X_test_scaled)

test_predictions

# Assuming that your predictions on the test set are stored in a 1-dimensional array called
# test_predictions. Feel free to modify this cell as long you create a CSV in the right format.

# Construct and save the submission:
submission_df = pd.DataFrame({
    "Id": test['id'],
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = "submission_{}.csv".format(timestamp)
submission_df.to_csv(filename, index=False)

print('Created a CSV file: {}.'.format("submission_{}.csv".format(timestamp)))
display(HTML("Download your test prediction <a href='" + filename + "' download>here</a>."))

y_train = y
y_scores = best_model.predict_proba(X_scaled)[:, 1]

fpr, tpr, thresholds = roc_curve(y_train, y_scores)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label="Logistic Regression (train)")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random guess")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

# Just run this cell, don't modify it.

print("spam: " + str(train.loc[5216]["spam"]))
print("\nemail:\n" + train.loc[5216]["email"])

# Just run this cell, don't modify it.

print("spam: " + str(train.loc[36]["spam"]))
print("\nemail:\n" + train.loc[36]["email"])

# Just run this cell, don't modify it.

print("spam: " + str(train.loc[1092]["spam"]))
print("\nemail:\n" + train.loc[1092]["email"])

# Simple model introduced at the start of this notebook. Just pay attention to the features.
some_words = ['drug', 'bank', 'prescription', 'memo', 'private']

X_train = words_in_texts(some_words, train['email'])
Y_train = np.array(train['spam'])

simple_model = LogisticRegression()
simple_model.fit(X_train, Y_train);

mask = (train['spam'] == 1) & train['email'].str.contains("bank", case=False, na=False)
train[mask]

email_idx = 27

prob_spam = simple_model.predict_proba(X_train)[:, 1]
initial_prob = prob_spam[email_idx]
print(f"\nPredicted probability of being spam: {np.round(initial_prob*100, 2)}%")
print("\nEmail:\n" + train.loc[email_idx]["email"])

feature_to_remove = "bank"

changed_words = some_words.copy()
changed_words.remove(feature_to_remove)

changed_model = LogisticRegression()
X_changed = words_in_texts(changed_words, train['email'])
y = train['spam']
changed_model.fit(X_changed, y)
changed_prob = changed_model.predict_proba(X_changed[[email_idx]])[:,1][0]

