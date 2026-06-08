# 🐦 End-to-End Sentiment Analysis

A machine learning pipeline that performs sentiment analysis on Twitter data, built with a fully reproducible DVC-managed workflow — from raw data ingestion through model evaluation.

---

## 📌 Overview

This project classifies the sentiment of tweets using a structured ML pipeline. Each stage is tracked and versioned with [DVC](https://dvc.org/), ensuring reproducibility and making it easy to re-run any part of the workflow independently.

**Pipeline stages:**

```
Raw Twitter Data
      │
      ▼
 Data Ingestion
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Extraction
      │
      ▼
 Model Building
      │
      ▼
Model Evaluation
```

---

## 🗂️ Project Structure

```
Sentiment-Analysis/
├── src/
│   └── Pipeline/
│       ├── data_ingestion.py       # Loads and saves raw Twitter data
│       ├── data_preprocessing.py   # Cleans and prepares text data
│       ├── feature_extraction.py   # Converts text to ML features
│       ├── model_building.py       # Trains the sentiment classifier
│       └── model_evaluation.py     # Evaluates and saves metrics
├── data/                           # Raw ingested data (DVC-tracked)
├── processed/                      # Cleaned data (DVC-tracked)
├── features/                       # Extracted features (DVC-tracked)
├── model/                          # Trained model artifacts (DVC-tracked)
├── metrices/                       # Evaluation metrics (DVC-tracked)
├── twitter.csv                     # Source Twitter dataset
├── dvc.yaml                        # DVC pipeline definition
├── dvc.lock                        # DVC pipeline lock file
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
├── .gitignore
└── .dvcignore
```

---

## ⚙️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| pandas / numpy | Data manipulation |
| scikit-learn | ML modelling & preprocessing |
| XGBoost | Gradient boosting classifier |
| DVC | Pipeline versioning & reproducibility |
| PyYAML | Configuration management |
| seaborn | Visualisation |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Dipam04/Sentiment-Analysis.git
cd Sentiment-Analysis
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

### 4. Initialize DVC

```bash
dvc init
```

### 5. Run the full pipeline

```bash
dvc repro
```

This will execute all stages in order: data ingestion → preprocessing → feature extraction → model building → model evaluation.

To run a specific stage only:

```bash
dvc repro <stage_name>
# e.g. dvc repro model_evaluation
```

---

## 🔄 DVC Pipeline Stages

| Stage | Script | Input | Output |
|---|---|---|---|
| `data_ingestion` | `data_ingestion.py` | `twitter.csv` | `data/` |
| `data_preprocessing` | `data_preprocessing.py` | `data/` | `processed/` |
| `feature_extraction` | `feature_extraction.py` | `processed/` | `features/` |
| `model_building` | `model_building.py` | `features/` | `model/` |
| `model_evaluation` | `model_evaluation.py` | `model/` | `metrices/` |

---

## 📊 Results

After running the pipeline, evaluation metrics are saved in the `metrices/` directory. You can view them with:

```bash
dvc metrics show
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Dipam** — [GitHub](https://github.com/Dipam04)
