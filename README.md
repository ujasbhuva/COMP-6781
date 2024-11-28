# AI-Generated Text Detection

## Overview
This project develops a classification system to differentiate between AI-generated and human-written text. The goal is to employ both traditional machine learning models and modern deep learning models to effectively classify text based on its origin. The system is trained on the **TweepFake** dataset, which consists of 25,572 tweets, equally divided between human-written and AI-generated content.

## Project Goal
To build a robust classification system that can distinguish AI-generated text from human-written text by leveraging machine learning (Random Forest and Multinomial Naive Bayes) and deep learning techniques (DistilBERT and ALBERT).

## Methodology

### Dataset
- **Dataset Used:** [TweepFake Dataset on Kaggle](https://www.kaggle.com/datasets/mtesconi/twitter-deep-fake-text)
- **Size:** 25,572 tweets (half AI-generated and half human-written).

### Data Preprocessing
- **Text Cleaning:** Removed special characters, URLs, and stopwords to reduce noise in the data.
- **Lemmatization:** Applied lemmatization to reduce words to their base or root form, improving generalization.
  
### Feature Extraction
- **TF-IDF (Term Frequency-Inverse Document Frequency):** Used to convert textual data into numerical features, capturing the importance of words in the corpus.

## Methods

### Baseline Models

1. **Random Forest Classifier**
   - Trained with 100 estimators and a max depth of 10.
   - Used GridSearchCV for hyperparameter tuning (`n_estimators`, `max_depth`, `min_samples_split`).

2. **Multinomial Naive Bayes**
   - Implemented a custom classifier with Laplace smoothing to handle zero probabilities.
   - Tuned hyperparameters for `alpha`, ranging from 0.1 to 0.9.

### Deep Learning Models

1. **DistilBERT**
   - Used the pre-trained **distilbert-base-uncased** model from Hugging Face.
   - Fine-tuned the model on the dataset using the Hugging Face `Trainer` with a learning rate of `1e-5`, batch size of 16, and 10 epochs.

2. **ALBERT**
   - Used the pre-trained **albert-base-v2** model from Hugging Face.
   - Fine-tuned similarly to DistilBERT with adjusted learning rate (`1e-5`), batch size of 8, and 10 epochs.

### Training
- Split the dataset into training, validation, and test sets.
- Trained models using the processed training data and validated using the validation set.
- Evaluated model performance using accuracy as the primary metric.

## Requirements
- Python 3.x
- Libraries:
  - scikit-learn
  - transformers (Hugging Face)
  - pandas
  - numpy
  - matplotlib (for visualization)
  - torch

## Installation

To install the necessary dependencies, run the following:

```bash
pip install -r requirements.txt
