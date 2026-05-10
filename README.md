# Quora Duplicate Question Detector

Detect whether two questions on Quora are asking the same thing using NLP, Machine Learning and Deep Learning.

## Results

| Model | Accuracy |
|-------|----------|
| LightGBM | 81.77% |
| BERT | 87.73% |

## Dataset
- Source: [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs)
- 400k question pairs
- 37% Duplicate, 63% Not Duplicate

## Tech Stack
- NLP: Word2Vec, Fuzzy Matching
- ML: LightGBM
- DL: BERT (HuggingFace)
- App: Streamlit

## Pretrained Model
Available on HuggingFace: [rakhikumari/quora-duplicate-bert](https://huggingface.co/rakhikumari/quora-duplicate-bert)

## How to Run

```bash
git clone https://github.com/your-username/quora-duplicate-detection.git
cd quora-duplicate-detection
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

├── data/
├── models/
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_feature_analysis_EDA.ipynb
│   ├── 04_ML_models.ipynb
│   └── 05_BERT.ipynb
├── app.py
├── requirements.txt
└── README.md