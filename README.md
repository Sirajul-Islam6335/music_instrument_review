# Music Instruments Review Analysis
The project aims to analyze customer reviews of musical instruments to understand sentiment patterns and build a model capable of classifying reviews based on text data.

## Text Preprocessing
Text preprocessing was applied to clean and standardize the raw data, including lowercasing, punctuation removal, tokenization, stopword removal, and lemmatization.
Missing text entries were also handled by replacing them with empty strings to ensure consistency for analysis.

## Model Development
## Naive Bayes (NB)
## Support Vector machine


## Evaluation & Results:
### The Multinomial Naive Bayes model performs:
It's achieving ahigh overall accuracy of 88.1%.
It shows strong recall (88.1%), indicating it correctly identifies most positive cases, though precision is slightly lower at 77.6%.
Overall, the F1-score of 82.6% reflects a balanced performance between precision and recall.

### The SVM model performs:
The SVM model demonstrates strong performance with an overall accuracy of 88.4%.
It achieves high precision (89.7%) and solid recall (88.4%), indicating reliable and consistent predictions.
The F1-score of 83.1% reflects a good balance between precision and recall.
# Results:
After deploying the user interface, the sentiment predictions are inconsistent, sometimes labeling positive reviews as negative or neutral.
This indicates that the model struggles with reliably distinguishing positive sentiments in user inputs.
Confusion Matrix:
The original dataset is highly imbalanced, with 1,809 positive, 101 negative, and 136 neutral samples.
Such imbalance can lead to biased model predictions, favoring the majority class.

## Oversampling & Retained model Evaluation:
After applying oversampling, all classes were balanced with 7,213 samples each, addressing the previous class imbalance.The retrained Multinomial Naive Bayes model achieved an accuracy of 75.9% and an F1-score of 79.8%, showing improved handling of minority classes.
The retrained SVM model performed better overall, with 83.3% accuracy and an F1-score of 83.6%, demonstrating robust and balanced predictions across all sentiment classes.

## Retrained model results & confusion matrix:
After retraining the models on the balanced dataset, the confusion matrix showed 1,809 correctly predicted positive, 94 neutral, and 60 negative samples.
The models now accurately identify positive and negative reviews, reflecting improved classification performance after oversampling.
However, some inconsistencies remain in detecting neutral sentiments, indicating overlap with other classes.
This suggests that further refinement of feature representation model tuning may be needed to enhance neutral sentiment recognition.

## Model tuning Results:
After model tuning, the SVM achieved improved performance with an accuracy of 88.4% and an F1-score of 83.2%, indicating better overall prediction capability.
Despite the performance gain, the issue with misclassifying neutral reviews persists, suggesting the model still struggles to distinguish neutral sentiments from positive or negative ones.


<img width="449" height="199" alt="retra" src="https://github.com/user-attachments/assets/56b3c0c4-348c-4238-aed5-656c77156f0d" />
