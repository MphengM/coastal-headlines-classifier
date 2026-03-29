# Coastal Headlines Classifier

A text classification pipeline for coastal and marine incident news, 
built using headlines sourced from the [GDELT Project](https://www.gdeltproject.org/).

This project is a companion to [coastal-news-classifier](https://github.com/MphengM/coastal-news-classifier),
which focuses on location extraction, geocoding, and interactive visualisation 
of coastal news data. This project focuses on text classification and model comparison.

## Project Overview

News headlines are classified into five coastal and marine incident categories 
using two approaches:

1. **TF-IDF + Linear SVM** — a supervised approach trained on labelled headlines
2. **Zero-shot classification** — using `facebook/bart-large-mnli` with no training data

Both experiments are tracked using MLflow, allowing direct comparison of results.

## Dataset

Headlines were collected via the GDELT DOC API using the `gdeltdoc` Python library,
covering January–February 2025. English-language results only.

| Category | Articles |
|---|---|
| Regulation & Policy | 516 |
| Red Tide | 447 |
| Pollution | 320 |
| Extreme Weather | 182 |
| Sea Life | 163 |
| **Total** | **1628** |

**Note:** Classification was performed on article titles rather than full text,
consistent with headline-based classification approaches in the literature and 
reflecting GDELT's data availability via the DOC API.

## Results

| Method | Accuracy |
|---|---|
| TF-IDF + Linear SVM | 81.29% |
| Zero-shot (bart-large-mnli) | 41.72% |

The supervised SVM outperforms zero-shot classification by ~40 percentage points,
demonstrating that domain-specific supervised learning remains highly competitive 
for specialised text classification tasks, even with a relatively small training set.

Both runs are logged in MLflow with full parameters and metrics.

## Tech Stack

- Python 3.13
- `gdeltdoc` — GDELT DOC API client
- `scikit-learn` — TF-IDF vectorization, SVM, evaluation metrics
- `transformers` (Hugging Face) — zero-shot classification pipeline
- `MLflow` — experiment tracking

## Repository Structure
```
coastal-headlines-classifier/
│
├── data/                          # Raw and processed CSV files
│   ├── extreme_weather_raw.csv
│   ├── pollution_raw.csv
│   ├── red_tide_raw.csv
│   ├── regulation_policy_raw.csv
│   └── sea_life_raw.csv
│
├── coastal_headlines_classifier.ipynb   # Main notebook
├── requirements.txt
└── README.md
```

## How to Run

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Launch Jupyter and open `coastal_headlines_classifier.ipynb`
4. To view MLflow experiment results:
```bash
mlflow ui
```
Then open `http://localhost:5000` in your browser.

## Future Work

- **Binary classifier** — filtering relevant vs irrelevant coastal headlines 
  as a preprocessing step
- **Full article text** — scraping article body text for richer features
- **Unsupervised clustering** — using LDA or k-means to validate category 
  boundaries, as in the original 2017 methodology
- **Separate conservation and illegal fishing** — currently merged into 
  `regulation_policy` due to low article counts; separating them may improve 
  classification precision
- **More descriptive candidate labels** — testing whether richer zero-shot 
  label descriptions improve transformer accuracy
- **BigQuery integration** — using GDELT via Google BigQuery for more efficient 
  bulk data collection

## Dependencies

See `requirements.txt` for full list. Key packages:
```
gdeltdoc
scikit-learn
transformers
mlflow
pandas
nltk
torch
```
