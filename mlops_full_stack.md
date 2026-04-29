# Complete MLOps Stack — HuggingFace + DagsHub + Evidently AI + GitHub Actions

---

## Tool-by-Tool Breakdown

---

### 1. 🤗 Hugging Face — Model Hub + Hosting + Inference

**What it is:** A platform to store, share, version, and serve ML models publicly or privately.

**3 things you can use it for:**

| Feature | What it does for you |
|---|---|
| **HF Hub (Model Registry)** | Push your `.pkl` models like Git — versioned, downloadable anywhere |
| **HF Spaces** | Deploy your FastAPI app as a live web app — FREE hosting |
| **HF Inference API** | Use pre-trained transformer models (e.g. `cardiffnlp/twitter-roberta-base-sentiment`) |

**For your project specifically:**

**Option A — Replace your sklearn model with a HF transformer:**
```python
from transformers import pipeline

# Zero-shot — no training needed
classifier = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

result = classifier("Customer was very interested but said budget is tight")
# Output: [{'label': 'neutral', 'score': 0.71}]
```

**Option B — Push your trained .pkl to HF Hub as artifact:**
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="app/models/sentiment_model_5000.pkl",
    path_in_repo="sentiment_model_5000.pkl",
    repo_id="your-username/sales-sentiment-model",
    repo_type="model"
)
```

**Option C — Deploy FastAPI app to HF Spaces (FREE hosting):**
```
# Create Space → Select "Docker" type
# Upload your Dockerfile + app/ folder
# HF builds and runs it automatically
# Gets URL: https://your-username-sentiment-api.hf.space
```

**Verdict for your use case:**
> 🟢 **Best for:** Free hosting (Spaces), model versioning (Hub), upgrading to transformer models
> 🔴 **Not for:** Heavy production traffic (Spaces has cold starts)

---

### 2. 📊 DagsHub — Git + DVC + MLflow in One Place

**What it is:** GitHub-like platform specifically built for ML — combines code, data, and experiments in one dashboard.

**What it gives you:**

| Feature | What it does |
|---|---|
| **Git repo hosting** | Like GitHub but ML-aware |
| **DVC remote storage** | Free 10GB for data/model versioning |
| **MLflow tracking server** | Hosted MLflow UI — no setup needed |
| **Experiment comparison** | Compare multiple training runs side-by-side |

**Setup (5 minutes):**
```bash
pip install dagshub mlflow dvc

# Connect to DagsHub
import dagshub
dagshub.init(repo_owner='your-username', repo_name='Sentiment_CICD', mlflow=True)

import mlflow
mlflow.set_tracking_uri("https://dagshub.com/your-username/Sentiment_CICD.mlflow")
```

**Add to your `model_trainer.py`:**
```python
import mlflow
import dagshub

dagshub.init(repo_owner='your-username', repo_name='Sentiment_CICD', mlflow=True)
mlflow.set_experiment("sales-sentiment")

with mlflow.start_run():
    mlflow.log_param("max_features", 5000)
    mlflow.log_param("ngram_range", "(1,3)")
    mlflow.log_param("best_C", grid_search.best_params_['C'])
    mlflow.log_param("best_solver", grid_search.best_params_['solver'])

    mlflow.log_metric("accuracy", float(accuracy))
    mlflow.log_metric("f1_weighted", float(cv_scores.mean()))
    mlflow.log_metric("f1_std", float(cv_scores.std()))

    # Log model
    mlflow.sklearn.log_model(best_model, "model")
    mlflow.log_artifact("app/models/tfidf_vectorizer_5000.pkl")
    mlflow.log_artifact("app/models/label_encoder_5000.pkl")
```

**DVC data versioning with DagsHub storage:**
```bash
dvc remote add origin https://dagshub.com/your-username/Sentiment_CICD.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user your-username
dvc remote modify origin --local password your-dagshub-token

dvc add app/data/
dvc push
```

**What the DagsHub dashboard looks like:**
```
DagsHub UI → Experiments Tab:
┌─────────────────────────────────────────────────────┐
│ Run Name          │ Accuracy │ F1    │ C   │ Solver │
│ run_20260428_0900 │ 0.8721   │ 0.864 │ 10  │ lbfgs  │
│ run_20260425_0600 │ 0.8544   │ 0.841 │ 1   │ lbfgs  │
│ run_20260422_0600 │ 0.8312   │ 0.819 │ 0.1 │ liblin │
└─────────────────────────────────────────────────────┘
```

**Verdict:**
> 🟢 **Best for:** Experiment tracking, data versioning, comparing runs — all FREE
> 🟢 **Replaces:** Self-hosted MLflow + S3 — zero infrastructure setup
> 🔴 **Not for:** App hosting / serving predictions

---

### 3. 📈 Evidently AI — Data & Model Monitoring

**What it is:** An open-source library that generates visual HTML reports about data drift, model performance degradation, and prediction quality.

**What it monitors for you:**

| Report Type | What it checks |
|---|---|
| **Data Drift** | Has the distribution of `Remarks` text changed? |
| **Target Drift** | Is the ratio of Positive/Negative/Neutral shifting? |
| **Model Performance** | Has accuracy/F1 dropped vs. reference period? |
| **Prediction Drift** | Are confidence scores getting lower over time? |

**Install:**
```bash
pip install evidently
```

**Add to your project — `app/drift_monitor.py`:**
```python
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.metrics import *
from datetime import datetime
import os

class DriftMonitor:
    def __init__(self, reference_data_path="app/data/reference.csv"):
        self.reference_data_path = reference_data_path
        self.reports_dir = "app/reports/drift"
        os.makedirs(self.reports_dir, exist_ok=True)

    def create_reference_dataset(self, df: pd.DataFrame):
        """Save first batch as reference baseline"""
        df[['Remarks', 'Sentiment', 'Confidence']].to_csv(
            self.reference_data_path, index=False
        )
        print(f"Reference dataset saved: {len(df)} records")

    def check_data_drift(self, current_df: pd.DataFrame) -> dict:
        """Compare current batch against reference"""
        if not os.path.exists(self.reference_data_path):
            print("No reference data — saving current as reference")
            self.create_reference_dataset(current_df)
            return {"status": "reference_created", "drift_detected": False}

        reference_df = pd.read_csv(self.reference_data_path)

        # Create Evidently report
        report = Report(metrics=[
            DataDriftPreset(),
            ColumnDriftMetric(column_name="Confidence"),
            ColumnDriftMetric(column_name="Sentiment"),
        ])

        report.run(
            reference_data=reference_df[['Confidence']],
            current_data=current_df[['Confidence']]
        )

        # Save HTML report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{self.reports_dir}/drift_report_{timestamp}.html"
        report.save_html(report_path)

        # Extract drift result
        result = report.as_dict()
        drift_detected = result['metrics'][0]['result']['dataset_drift']

        return {
            "drift_detected": drift_detected,
            "report_path": report_path,
            "timestamp": timestamp
        }

    def check_model_performance(self, df_with_predictions: pd.DataFrame) -> dict:
        """Check if model quality has degraded"""
        report = Report(metrics=[
            ClassificationPreset(),
        ])

        # Need both prediction and ground truth
        if 'Sentiment' not in df_with_predictions.columns:
            return {"status": "no_labels_available"}

        ref = pd.read_csv(self.reference_data_path)

        report.run(
            reference_data=ref,
            current_data=df_with_predictions[['Sentiment', 'Confidence']]
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{self.reports_dir}/performance_report_{timestamp}.html"
        report.save_html(report_path)

        return {"report_path": report_path, "status": "generated"}
```

**Wire into your `main.py` batch endpoint:**
```python
from drift_monitor import DriftMonitor
drift_monitor = DriftMonitor()

@app.post("/batch_analyze")
async def batch_analyze_api(file: UploadFile = File(...)):
    # ... your existing batch analysis code ...

    # After predictions are added to df:
    drift_result = drift_monitor.check_data_drift(df)
    
    if drift_result.get("drift_detected"):
        print("DRIFT DETECTED — scheduling retraining")
        # Optionally trigger retraining here

    return JSONResponse(content={
        "success": True,
        "data": { ... },
        "drift_check": drift_result
    })
```

**Add a drift report endpoint:**
```python
@app.get("/api/drift-report")
async def get_drift_report():
    reports = sorted(os.listdir("app/reports/drift"), reverse=True)
    if not reports:
        return {"status": "no_reports"}
    latest = reports[0]
    return FileResponse(f"app/reports/drift/{latest}", media_type="text/html")
```

**Verdict:**
> 🟢 **Best for:** Visual drift reports, data quality checks after each batch upload
> 🟢 **Free:** Fully open source, runs locally, generates HTML reports
> 🔴 **Not for:** Real-time streaming monitoring (needs batch data)

---

### 4. ⚙️ GitHub Actions — Full CI/CD Orchestrator

**What it is:** The automation backbone that ties everything together.

**Your complete workflow chain:**

```
Push CSV → data-monitor.yml
              ↓
         model-training.yml  (DagsHub MLflow logs run)
              ↓
         quality-gate.yml    (check accuracy/F1 thresholds)
              ↓ PASS
         docker-build.yml    (build image, push to GHCR or HF)
              ↓
         deploy.yml          (deploy to HF Spaces / Render / EC2)
              ↓
         monitor.yml         (daily Evidently drift check)
```

**Complete `model-training.yml` with DagsHub:**
```yaml
name: Automated Model Training

on:
  schedule:
    - cron: '0 2 */3 * *'
  workflow_dispatch:
  push:
    paths:
      - 'app/data/**'

jobs:
  train-model:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install mlflow dagshub dvc

      - name: Pull latest data with DVC
        env:
          DAGSHUB_TOKEN: ${{ secrets.DAGSHUB_TOKEN }}
        run: |
          dvc remote modify origin --local user ${{ secrets.DAGSHUB_USER }}
          dvc remote modify origin --local password ${{ secrets.DAGSHUB_TOKEN }}
          dvc pull

      - name: Download NLTK data
        run: |
          python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('vader_lexicon')"

      - name: Train model with MLflow tracking
        env:
          DAGSHUB_USER: ${{ secrets.DAGSHUB_USER }}
          DAGSHUB_TOKEN: ${{ secrets.DAGSHUB_TOKEN }}
          MLFLOW_TRACKING_URI: https://dagshub.com/${{ secrets.DAGSHUB_USER }}/Sentiment_CICD.mlflow
          MLFLOW_TRACKING_USERNAME: ${{ secrets.DAGSHUB_USER }}
          MLFLOW_TRACKING_PASSWORD: ${{ secrets.DAGSHUB_TOKEN }}
          GITHUB_ACTIONS: "true"
        run: |
          cd app && python model_trainer.py

      - name: Upload models and reports
        uses: actions/upload-artifact@v4
        with:
          name: trained-models
          path: |
            app/models/*.pkl
            app/models/training_metadata.json
            app/reports/**
          retention-days: 30
```

**`quality-gate.yml`:**
```yaml
name: Quality Gate

on:
  workflow_run:
    workflows: ["Automated Model Training"]
    types: [completed]

jobs:
  gate:
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    runs-on: ubuntu-latest
    outputs:
      passed: ${{ steps.check.outputs.passed }}
    steps:
      - uses: actions/checkout@v4

      - name: Download models
        uses: actions/download-artifact@v4
        with:
          name: trained-models
          path: app/models/
          run-id: ${{ github.event.workflow_run.id }}
          github-token: ${{ secrets.GITHUB_TOKEN }}

      - name: Run quality gate
        id: check
        run: |
          python -c "
          import json, sys
          with open('app/models/training_metadata.json') as f:
              meta = json.load(f)
          acc = meta['accuracy']
          f1  = meta['cv_f1_score']
          print(f'Accuracy={acc:.4f}  F1={f1:.4f}')
          passed = acc >= 0.75 and f1 >= 0.70
          print('PASSED' if passed else 'FAILED')
          sys.exit(0 if passed else 1)
          "
          echo "passed=true" >> $GITHUB_OUTPUT

      - name: Create Issue on failure
        if: failure()
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: '🚨 Model Quality Gate Failed',
              body: 'Accuracy or F1 below threshold. Check the Actions run for details.',
              labels: ['mlops', 'model-quality']
            })
```

---

## 5. Deployment Options Compared

| Platform | Cost | Setup | Best For | Limitations |
|---|---|---|---|---|
| **HF Spaces (Docker)** | FREE | Easy | Demo + testing | Cold starts, 16GB RAM |
| **Render** | FREE tier | Easy | Production hosting | Sleeps after 15min inactivity |
| **Railway** | $5/mo | Easy | Always-on production | Paid |
| **AWS EC2** | ~$5-10/mo | Medium | Full control | Needs DevOps knowledge |
| **Google Cloud Run** | Pay-per-use | Medium | Scalable production | GCP account needed |

### Recommendation for Your Project:

```
DEVELOPMENT / TESTING  →  HF Spaces (free, instant URL, shareable)
PRODUCTION / ALWAYS-ON →  Render Free Tier or Railway ($5/mo)
ENTERPRISE / SCALE     →  AWS EC2 or Google Cloud Run
```

---

## 6. Complete Integrated Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    YOUR COMPLETE MLOPS STACK                        │
│                                                                     │
│  DATA              TRAINING            DEPLOYMENT      MONITORING   │
│  ────              ────────            ──────────      ──────────   │
│                                                                     │
│  app/data/    →   model_trainer.py  →  HF Spaces     → Evidently   │
│  (DVC)            (MLflow logs      →  Render         → AI reports  │
│  DagsHub          to DagsHub)       →  Railway                     │
│  storage                                                            │
│                                                                     │
│  GitHub       →   GitHub Actions   →  Docker         → GitHub      │
│  (code)           (orchestrates        GHCR image      Actions     │
│                    everything)                          (daily      │
│                                                         monitor)    │
└─────────────────────────────────────────────────────────────────────┘
```

### Tool Responsibility Map:

| Tool | Responsibility |
|---|---|
| **GitHub** | Code versioning, CI/CD trigger |
| **DagsHub** | Data versioning (DVC) + Experiment tracking (MLflow UI) |
| **GitHub Actions** | Automates: train → gate → build → deploy → monitor |
| **HuggingFace Hub** | Model artifact registry (versioned .pkl storage) |
| **HuggingFace Spaces** | Free app hosting (FastAPI Docker app) |
| **Evidently AI** | Drift reports after every batch upload |
| **Docker + GHCR** | Reproducible container builds |

---

## 7. Step-by-Step Setup Order

```
Week 1:
  1. Create DagsHub account → link your GitHub repo
  2. Add dagshub.init() to model_trainer.py
  3. Push data to DagsHub DVC remote
  4. Run training → see experiments in DagsHub dashboard

Week 2:
  5. Add quality-gate.yml to GitHub Actions
  6. Add Evidently drift check to batch_analyze endpoint
  7. Write Dockerfile → test locally

Week 3:
  8. Create HF Space (Docker type) → deploy FastAPI
  9. Add docker-build.yml to push to GHCR
  10. Add deploy.yml to push to HF Spaces or Render

Week 4:
  11. Add monitor.yml for daily drift check
  12. Wire Evidently HTML reports to /api/drift-report endpoint
  13. Set up GitHub Secrets for all tokens
```

---

## 8. GitHub Secrets Needed

| Secret | Where to get it |
|---|---|
| `DAGSHUB_USER` | Your DagsHub username |
| `DAGSHUB_TOKEN` | DagsHub → Settings → Tokens |
| `HF_TOKEN` | HuggingFace → Settings → Access Tokens |
| `HF_SPACE_NAME` | e.g. `your-username/sentiment-api` |
| `RENDER_API_KEY` | Render → Account → API Keys |
| `RENDER_SERVICE_ID` | Render → Your Service → Settings |

---

## 9. Final Verdict — What to Use for What

| Need | Best Tool |
|---|---|
| **Experiment tracking** | DagsHub (free MLflow server) |
| **Data versioning** | DagsHub (free DVC remote 10GB) |
| **Model registry** | DagsHub + HuggingFace Hub |
| **CI/CD automation** | GitHub Actions |
| **Data drift reports** | Evidently AI |
| **Free app hosting/demo** | HuggingFace Spaces |
| **Production hosting** | Render (free) or Railway ($5/mo) |
| **Container build** | Docker + GitHub Container Registry |

> [!IMPORTANT]
> **Minimum viable MLOps (start here):**
> DagsHub (MLflow + DVC) + GitHub Actions (quality gate) + HuggingFace Spaces (hosting)
> — All FREE, setup in 1 week, covers 80% of MLOps needs.

> [!TIP]
> **Evidently AI** is your secret weapon — after every CSV batch upload in your app, run a drift report automatically. If Evidently detects drift → GitHub Actions triggers retraining. This creates a fully autonomous self-healing ML system.
