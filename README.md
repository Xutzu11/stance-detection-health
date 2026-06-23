# Stance Detection in Controversial Health Discourses

Fine-tuning BERTweet for stance classification on health-related tweets, with classical ML baselines and cross-domain transfer experiments.

## Repository Structure

```
├── Project/
│   ├── data_loading_eda.ipynb       # Notebook 1 – data loading & EDA
│   ├── baselines.ipynb              # Notebook 2 – TF-IDF + LR/SVM baselines
│   ├── bertweet_finetuning.ipynb    # Notebook 3 – BERTweet fine-tuning
│   ├── analysis.ipynb               # Notebook 4 – results, transfer & error analysis
│   ├── plots/                       # Generated charts (PNG)
│   └── results/                     # Generated result tables (CSV/JSON)
├── paper/
│   └── Stance Detection in Controversial Health Discourses.pdf
├── presentation/
│   └── Stance Detection Health Discourses.pptx
└── README.md
```

## Datasets

| Dataset | Source | Classes | Size |
|---|---|---|---|
| `covid-vax-stance` | [Poddar et al., ICWSM 2022](https://github.com/sohampoddar26/covid-vax-stance) | ProVax / AntiVax / Neutral | 3,139 tweets |
| `tweeteval-stance_abortion` | [Cardiff NLP / SemEval-2016](https://huggingface.co/datasets/cardiffnlp/tweet_eval) | favor / against / none | 933 tweets |

Both datasets are mapped to a unified label set: **FAVOR / AGAINST / NONE**.

## Model

- **BERTweet** (`vinai/bertweet-base`) — RoBERTa-based model pre-trained on 850 M English tweets.
- Fine-tuned separately on each dataset (10 epochs, batch size 32, lr 2e-5, weighted cross-entropy).

## Results (Macro-F1)

| Model | covid-vax-stance | TweetEval-stance |
|---|---|---|
| Logistic Regression (TF-IDF) | 0.3955 | 0.3719 |
| SVM (TF-IDF) | 0.4931 | 0.5110 |
| **BERTweet (fine-tuned)** | **0.6240** | **0.6615** |

Cross-domain transfer (no retraining): BERTweet CVS to TweetEval = 0.2696, TweetEval to CVS = 0.2970.

## Setup & Running on Google Colab

### Prerequisites

- A Google account with Google Drive.
- A Google Colab session (free tier is sufficient for Notebooks 1 & 2; **Notebook 3 requires a GPU runtime** – use *Runtime / Change runtime type / T4 GPU*).

### 1. Prepare your Google Drive

Create the following directory tree inside **My Drive**. Everything the notebooks read or write lives here.

```
My Drive/
└── StanceProject/
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── plots/
    ├── results/
    ├── predictions/
    └── models/
```

You can create `StanceProject/` manually in Google Drive and the sub-folders will be created automatically when you run the notebooks for the first time.

### 2. Upload the notebooks to Colab

Option A – upload directly in Colab:
1. Go to [colab.research.google.com](https://colab.research.google.com).
2. *File / Upload notebook* and upload the four `.ipynb` files from the `Project/` folder.

Option B – open from Drive:
1. Copy the four notebooks into any folder in your Drive (e.g. `My Drive/StanceProject/notebooks/`).
2. Right-click a notebook in Drive, then *Open with / Google Colaboratory*.

### 3. Run the notebooks in order

> **Important:** run them sequentially – each notebook depends on outputs from the previous one.

#### Notebook 1 – `data_loading_eda.ipynb`
- **Runtime:** CPU (no GPU needed).
- **What it does:** Downloads the `covid-vax-stance` CSVs from GitHub and the `tweeteval-stance_abortion` split from Hugging Face, cleans and unifies labels, and produces EDA charts (label distributions, text-length histograms, word clouds).
- **Outputs written to Drive:**
  - `StanceProject/data/raw/` – 3 raw CSVs
  - `StanceProject/data/processed/covid_vax_stance.csv`
  - `StanceProject/data/processed/tweeteval_stance.csv`
  - `StanceProject/plots/label_distribution.png`
  - `StanceProject/plots/text_length.png`
  - `StanceProject/plots/wordcloud_covid_vax_stance.png`
  - `StanceProject/plots/wordcloud_tweeteval_stance_abortion.png`
  - `StanceProject/results/dataset_summary.csv`

#### Notebook 2 – `baselines.ipynb`
- **Runtime:** CPU.
- **What it does:** Trains TF-IDF + Logistic Regression and TF-IDF + LinearSVC on both datasets. Evaluates in-domain and cross-domain (zero-shot transfer).
- **Outputs written to Drive:**
  - `StanceProject/plots/cm_covid-vax-stance_lr.png`
  - `StanceProject/plots/cm_covid-vax-stance_svm.png`
  - `StanceProject/plots/cm_tweeteval-stance_lr.png`
  - `StanceProject/plots/cm_tweeteval-stance_svm.png`
  - `StanceProject/results/baseline_results.csv`

#### Notebook 3 – `bertweet_finetuning.ipynb`
- **Runtime:** GPU (T4 recommended). Enable via *Runtime / Change runtime type / T4 GPU*.
- **What it does:** Fine-tunes `vinai/bertweet-base` for stance classification. **Must be run twice**:
  1. Set `DATASET = 'cvs'` at the top of the notebook → trains on `covid-vax-stance`.
  2. Re-open / restart, set `DATASET = 'te'` → trains on `tweeteval-stance_abortion`.
- **Outputs written to Drive (per run):**
  - `StanceProject/models/bertweet_cvs/` – best checkpoint (~500 MB; **not committed to git**)
  - `StanceProject/models/bertweet_te/` – best checkpoint
  - `StanceProject/results/bertweet_cvs_training_log.json`
  - `StanceProject/results/bertweet_te_training_log.json`
  - `StanceProject/plots/cm_bertweet_cvs.png`
  - `StanceProject/plots/cm_bertweet_te.png`
  - `StanceProject/predictions/bertweet_cvs_preds.csv`
  - `StanceProject/predictions/bertweet_te_preds.csv`
- **Note:** The saved model checkpoints are large (~500 MB each) and are excluded from this repository via `.gitignore`. They persist in your Google Drive across Colab sessions.

#### Notebook 4 – `analysis.ipynb`
- **Runtime:** GPU (needed for cross-domain BERTweet inference).
- **What it does:** Loads all baseline and BERTweet results, runs cross-domain transfer inference with the saved BERTweet checkpoints, performs error analysis, and generates publication-quality comparison charts.
- **Inputs required:** all outputs from Notebooks 1–3 must exist in Drive.
- **Outputs written to Drive:**
  - `StanceProject/plots/model_comparison.png`
  - `StanceProject/plots/per_class_heatmap.png`
  - `StanceProject/plots/learning_curve.png`
  - `StanceProject/results/bertweet_cvs_errors.csv`
  - `StanceProject/results/final_results.csv`

### 4. Copy results to the repository

After running all notebooks, copy the generated files from your Drive into the corresponding local folders before committing:

- `StanceProject/plots/` into `Project/plots/`
- `StanceProject/results/` into `Project/results/`

Model checkpoints (`StanceProject/models/`) are intentionally **excluded** – they exceed GitHub's file size limits.

## Dependencies

All dependencies are installed inside each Colab notebook via `!pip install`. Key packages:

| Package | Purpose |
|---|---|
| `transformers` | BERTweet tokenizer & model |
| `torch` | PyTorch backend |
| `accelerate` | Mixed-precision training |
| `datasets` | Loading TweetEval from Hugging Face |
| `scikit-learn` | TF-IDF, LR, SVM, metrics |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Plotting |
| `wordcloud` | Word cloud visualisation |
| `emoji==0.6.0` | BERTweet emoticon normalisation |

## Citation

If you use this work, please cite the datasets:

```
@inproceedings{poddar2022covid,
  title={Winds of Change: Impact of COVID-19 on Vaccine-Related Opinions of Twitter Users},
  author={Poddar, Soham and others},
  booktitle={ICWSM},
  year={2022}
}

@inproceedings{barbieri2020tweeteval,
  title={TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification},
  author={Barbieri, Francesco and others},
  booktitle={EMNLP Findings},
  year={2020}
}
```
