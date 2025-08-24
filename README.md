# MLOps End-to-End Pipeline

This repository demonstrates a complete ML workflow managed with **DVC**: ingest an SMS spam dataset, clean & pre-process text, extract **TF-IDF** features, train a **RandomForest** classifier, evaluate it, and log experiments—reproducibly. It showcases versioned data, tracked parameters/metrics, and experiment reproduction via DVC pipelines.&#x20;

---

## 🚀 Quick Overview

* **Goal:** Build an end-to-end, reproducible spam-detection pipeline with DVC.
* **Pipeline stages:**

  * **Data Ingestion:** Download dataset, rename columns, split train/test.
  * **Pre-processing:** Label-encode target, remove duplicates, transform messages (lowercase, tokenize, remove stop-words/punctuation, stem).
  * **Feature Engineering:** Apply **TF-IDF**; control vocabulary size via params.
  * **Model Training:** Train **RandomForest** with hyper-params from `params.yaml`; save `model.pkl`.
  * **Model Evaluation:** Compute **accuracy, precision, recall, AUC**; log metrics with **dvclive** to `reports/metrics.json`.
* **Tech Stack:** Python, pandas, NumPy, NLTK, scikit-learn, **DVC**, **dvclive**, YAML.
* **Highlights:**

  * **DVC pipeline:** `dvc.yaml` defines stages, dependencies, outputs, parameters.
  * **Parameters & experiments:** `params.yaml` stores tunables (e.g., `test_size`, `max_features`, `n_estimators`, `random_state`); dvclive logs runs & metrics.
  * **Logging & testing:** Scripts log to `logs/` and validate inputs to avoid common errors.&#x20;

---

## 📂 Repository Structure

| Path                                         | Purpose                                                |   |
| -------------------------------------------- | ------------------------------------------------------ | - |
| `dvc.yaml`                                   | Pipeline stages, deps, outs, params                    |   |
| `params.yaml`                                | Tunables: split size, TF-IDF features, RF hyper-params |   |
| `src/1_Data_Ingestion.py`                    | Download & split dataset                               |   |
| `src/2_Pre_Processing.py`                    | Encode labels; clean & stem text                       |   |
| `src/3_Feature_Engineering.py`               | TF-IDF feature matrices (train/test)                   |   |
| `src/4_Model_Training.py`                    | Train RF using `params.yaml`; save `model.pkl`         |   |
| `src/5_Model_Evaluation.py`                  | Evaluate & log metrics (dvclive)                       |   |
| `data/`                                      | Raw, interim, processed data                           |   |
| `models/`                                    | Trained models                                         |   |
| `reports/`                                   | Generated reports (e.g., `metrics.json`)               |   |
| `Experiments/Experimentation_Notebook.ipynb` | Dev/experimentation notebook                           |   |
| `Project_Flow.txt`                           | Step-by-step DVC setup & run guide                     |   |
| `requirements.txt`                           | Python dependencies                                    |   |

---

## ⚙️ How to Run the Pipeline

### 1) Clone & install

```bash
git clone https://github.com/PrakashD2003/MLOPs-End-To-Pipeline.git
cd MLOPs-End-To-Pipeline

# (optional) create & activate venv
python3 -m venv venv
source venv/bin/activate   # on Windows: .\.venv\Scripts\activate

# install deps
pip install -r requirements.txt
```

### 2) Initialize DVC (first time)

```bash
dvc init
```

### 3) Reproduce the pipeline

```bash
# runs all stages in dvc.yaml:
# ingest → preprocess → features → train → evaluate
dvc repro
```

### 4) Track experiments (dvclive / DVC exp)

```bash
# view experiment table
dvc exp show

# remove or apply experiments
dvc exp remove <exp-name>
dvc exp apply  <exp-name>
```

### 5) Tune hyper-parameters

Edit `params.yaml` (e.g., `n_estimators`, `test_size`, `max_features`), then:

```bash
dvc repro
```

DVC re-runs only affected stages using cached results for fast iteration.&#x20;

---

## 📊 Key Learnings

* **Data versioning & pipelines:** Use DVC to version datasets and reproduce multi-stage workflows.
* **Parameter management:** Centralize hyper-params in YAML for quick tuning.
* **Text processing:** Clean & stem messages before modeling.
* **Feature extraction:** Convert text to vectors with TF-IDF; control vocabulary size.
* **RandomForest modeling:** Train & evaluate a robust baseline classifier.
* **Experiment tracking:** Log metrics/params with dvclive; compare runs via `dvc exp`.&#x20;

---
## 🙌 Closing Notes

A practical guide to building a reproducible ML workflow with DVC—covering ingestion, processing, feature engineering, training, evaluation, and experiment tracking. Clone, modify, and extend—contributions welcome!&#x20;
