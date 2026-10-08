# Sentiment Analysis of Musical Instrument Reviews

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-154F5B)
![Gradio](https://img.shields.io/badge/Gradio-app-FF7C00?logo=gradio&logoColor=white)

This project predicts whether an Amazon review of a musical instrument or accessory is **positive**, **neutral** or **negative**, based only on what the reviewer wrote. The final model is served in a small Gradio app: you type in a review and it shows how likely each sentiment is.

## Why I rebuilt this project

My first version of this project reported **88% accuracy** and a cross-validation score of **0.9996**. Both numbers looked great, and both were misleading:

- 88% of the reviews are positive, so a model that always answers "positive" gets 88% accuracy without learning anything. My Naive Bayes model was doing exactly that.
- I had oversampled the data *before* cross-validation, so copies of the same review ended up in both the training and validation folds. That leakage is where the 0.9996 came from.

This version fixes those mistakes. The scores are lower, but they reflect how the model actually behaves on reviews it has never seen.

## Results

Evaluated once, at the end, on a held-out test set of 2,053 reviews (20% of the data, stratified).

| Metric | First version | Final model |
|---|---|---|
| Macro F1 | 0.509 | **0.611** |
| Balanced accuracy | 0.502 | **0.643** |
| Accuracy | 0.843 | **0.864** |
| Neutral recall | 25% | **46%** |
| Negative recall | 33% | **55%** |

Per-class results for the final model:

| Class | Precision | Recall | F1 | Reviews in test set |
|---|---|---|---|---|
| positive | 0.953 | 0.915 | 0.933 | 1,805 |
| neutral | 0.341 | 0.465 | 0.393 | 155 |
| negative | 0.468 | 0.548 | 0.505 | 93 |

I use **macro F1** as the main metric because it gives the small neutral and negative classes the same weight as the large positive one. A model can't score well on it by ignoring the minority classes.

## Dataset

10,261 Amazon reviews from the Musical Instruments category ([Julian McAuley, UCSD](https://jmcauley.ucsd.edu/data/amazon/)). Each review has a text, a short summary (the review title) and a 1–5 star rating.

I turned the star ratings into three labels:

| Stars | Label | Reviews | Share |
|---|---|---|---|
| 4–5 | positive | 9,022 | 87.9% |
| 3 | neutral | 772 | 7.5% |
| 1–2 | negative | 467 | 4.6% |

The data is mostly clean. Only 7 reviews have no text, and all of them still have a summary.

## Approach

### 1. Text cleaning that keeps negations

NLTK's standard English stop-word list removes words like *not*, *no* and *don't*. That's a problem for sentiment: my old cleaning turned "This tuner does not work" into "tuner work".

Negation words also turn out to be a useful signal in this data. They appear in 83% of negative reviews, 72% of neutral ones and 59% of positive ones.

The new cleaning function:

- lowercases the text and removes URLs, numbers and punctuation
- expands contractions ("wouldn't" → "would not", "can't" → "can not")
- removes stop words, but keeps *no*, *not*, *nor* and *never*
- lemmatizes each word with NLTK's WordNet lemmatizer

| Original | Old cleaning | New cleaning |
|---|---|---|
| This tuner does not work at all. | tuner work | tuner not work |
| I wouldn't recommend these strings. | wouldnt recommend string | would not recommend string |
| No hum, no noise, great cable! | hum noise great cable | no hum no noise great cable |

### 2. Adding the review summary

Summaries such as "Great cable" or "Broke in a week" carry a lot of sentiment in very few words, so I put the summary in front of the review text before cleaning.

### 3. Leak-free evaluation

- Stratified 80/20 train/test split. The test set is used only once, at the very end.
- All model selection uses 5-fold stratified cross-validation on the training set.
- TF-IDF, resampling and the classifier all sit inside one scikit-learn pipeline, so each fold only ever learns from its own training data.

### 4. Model comparison

5-fold cross-validation on the training set, sorted by macro F1:

| Model | Macro F1 | Neutral F1 | Negative F1 | Accuracy |
|---|---|---|---|---|
| **Logistic regression + class weights** | **0.584** | **0.354** | **0.461** | 0.870 |
| Logistic regression + oversampling | 0.578 | 0.337 | 0.454 | 0.882 |
| Linear SVM + class weights | 0.515 | 0.254 | 0.344 | 0.892 |
| First version: oversampling + SVM | 0.474 | 0.214 | 0.289 | 0.841 |
| Complement Naive Bayes | 0.316 | 0.000 | 0.011 | 0.880 |
| Always predict "positive" | 0.312 | 0.000 | 0.000 | 0.879 |
| First version: Naive Bayes | 0.312 | 0.000 | 0.000 | 0.879 |

Plain Naive Bayes scores exactly the same as always predicting "positive". That's where my original 88% came from. The linear SVM has the highest accuracy, but it finds far fewer neutral and negative reviews than logistic regression.

### 5. What each change contributed

To see which changes actually helped, I added them one at a time, using logistic regression with class weights each time:

| Step | Macro F1 |
|---|---|
| Old cleaning, review text only, single words | 0.485 |
| + review summary | 0.536 |
| + word pairs (bigrams), `min_df=2`, sublinear TF | 0.572 |
| + negations kept | 0.584 |

Adding the summary gave the biggest jump. Keeping negations helped less than I expected (about +0.01), probably because negative reviews usually contain other clear clues such as *broke* or *returned*.

### 6. Tuning

A grid search over n-gram range, `min_df` and the regularisation strength `C` picked:

- TF-IDF with single words and word pairs, `min_df=2`, sublinear TF
- Logistic regression with `class_weight="balanced"` and `C=0.5`

That gives a cross-validated macro F1 of 0.585. The top settings were all very close to each other, so tuning made little difference.

## What the model learned

The words with the largest weights for each class make sense:

- **Positive:** *great*, *perfect*, *love*
- **Neutral:** *ok*, *okay*, *not bad*. Some reviewers even write "three stars" in their summary, and the model picked that up.
- **Negative:** *not*, *broke*, *not work*, *defective*

## Error analysis

Neutral is still the weakest class. The most common mistake is a positive review that mentions a complaint and gets labelled neutral.

Many of the model's most confident mistakes are reviews where the text and the star rating disagree, for example an enthusiastic review with only 3 stars. No text-only model can get those right. Most of the remaining mistakes are mixed reviews ("good, but…"), which a bag-of-words model struggles with.

## The app

The notebook ends with a Gradio app. You enter an optional review title and the review text, and it shows the probability of each sentiment.

Some examples from the notebook:

| Review | Prediction |
|---|---|
| "Doesn't work with my amp at all. Would not recommend." | negative (86%) |
| "It's okay. Sound is fine, nothing special." | neutral (83%) |
| "Love it" + "Best strings I've used, bright tone and they stay in tune." | positive (92%) |

## Project structure

```
music_instrument_review/
├── music_instruments_review.ipynb    # full analysis, training and Gradio app
├── Musical_instruments_reviews.csv   # dataset
├── sentiment_model.joblib            # trained pipeline (created by the notebook)
└── README.md
```

## How to run it

```bash
git clone https://github.com/Sirajul-Islam6335/music_instrument_review.git
cd music_instrument_review

pip install numpy pandas matplotlib seaborn nltk scikit-learn imbalanced-learn joblib gradio jupyter

jupyter notebook music_instruments_review.ipynb
```

Run the cells from top to bottom. The NLTK data downloads automatically on the first run. The last cell launches the app at `http://127.0.0.1:7860`.

### Using the saved model

`sentiment_model.joblib` contains the full TF-IDF + logistic regression pipeline, refit on all the data. It expects text that has already gone through the same cleaning as in training, so use the `combine()` and `clean_text()` functions from the notebook:

```python
import joblib

model = joblib.load("sentiment_model.joblib")
text = clean_text(combine("Broke in a week", "The jack came loose and now it crackles."))
print(model.predict([text])[0])  # negative
```

## Limitations

- **Neutral reviews are hard** (F1 ≈ 0.39). Three-star reviews often mix praise and complaints.
- **Some labels are noisy.** The label comes from the star rating, and sometimes the rating doesn't match the text.
- **Bag-of-words has no sense of word order.** In a sentence like "I can't believe how good this sounds", the model sees *not* and *good* but can't tell how they relate.
- **The data is narrow.** It covers one product category, and all reviews were written before 2015.

## Next steps

- Fine-tune a transformer model such as DistilBERT, which should handle mixed and sarcastic reviews better.
- Find and review cases where the star rating clearly contradicts the text.
- Deploy the app on Hugging Face Spaces so it can be tried without installing anything.

## Author

**Sirajul Islam** · [GitHub](https://github.com/Sirajul-Islam6335)
