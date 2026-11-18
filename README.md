# Credit Ledger: credit risk scoring with SHAP explanations

A FastAPI service and web form that scores consumer loan applications for
default risk with a calibrated XGBoost model. Each score comes with a SHAP
breakdown of the factors behind it.

## Project layout

| Path | Purpose |
|---|---|
| `credit_risk/config.py` | Column lists and allowed category values, shared by training and serving |
| `credit_risk/data.py` | Loading and cleaning the raw CSV |
| `credit_risk/train.py` | Training, calibration, threshold selection, evaluation |
| `credit_risk/model.py` | Model artifact: load/save, predict, SHAP explanations |
| `credit_risk/tracking.py` | MLflow logging and model registration (on when `MLFLOW_TRACKING_URI` is set) |
| `credit_risk/plots.py` | Calibration, precision-recall and SHAP importance plots logged to MLflow |
| `infra/app.py` | AWS CDK stack: S3 bucket, ECR repo, IAM roles, SageMaker serverless MLflow App |
| `scripts/launch_training.py` | Uploads data, builds and pushes the training image, runs the SageMaker job |
| `scripts/promote_model.py` | Marks a registered version as `champion` (MLflow alias + SageMaker approval) |
| `scripts/fetch_model.py` | Downloads a registered version's model files |
| `scripts/deploy_serving.py` | Deploys the champion model to Lambda and smoke-tests the live URL |
| `.github/workflows/` | CI (tests, CDK synth) and the Lambda deploy |
| `credit_risk/monitoring.py` | Reference profile of the training data, PSI |
| `credit_risk/evaluate.py`, `credit_risk/register.py` | Pipeline steps: compare with champion, register for approval |
| `monitoring/drift_check.py` | Daily drift-check Lambda |
| `infra/retraining.py`, `infra/monitoring.py` | CDK: retraining pipeline; drift check, alarms, dashboard |
| `scripts/simulate_traffic.py`, `scripts/start_retraining.py`, `scripts/subscribe_alerts.py` | Demo and setup helpers |
| `Dockerfile.train` | Training image (same pinned libraries as the serving image) |
| `main.py` | FastAPI app (`/health`, `/predict`, static frontend) |
| `static/` | Web form |
| `artifacts/` | Trained model, written by `train.py` |
| `tests/` | Model and API tests |
| `Credit_Risk.ipynb` | Original exploration notebook (training now lives in `credit_risk/train.py`) |

## Setup

Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Train

```powershell
.\.venv\Scripts\python.exe -m credit_risk.train --data credit_risk_dataset.csv --out artifacts
```

Takes about 1.5 minutes with the default 60 search trials (`--n-iter`). Writes:

- `preprocessor.joblib`: fitted scikit-learn preprocessing
- `xgb_model.json`: XGBoost booster in its version-stable JSON format
- `model_meta.json`: calibration, thresholds, data hash, best parameters, library versions
- `metrics.json`: test-set metrics, plus a logistic regression baseline

How the training avoids leakage:

1. A stratified 20% test set is held out and only used for the final report.
2. Hyperparameters are tuned with 5-fold CV on the training set (PR-AUC).
3. Platt calibration is fitted on out-of-fold log-odds from the training set.
4. The decision threshold maximises F1 on the calibrated out-of-fold probabilities.
5. The Medium Risk band starts at the training default rate (about 22%).

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The tests check, among other things, that the SHAP factors add up to the
model's log-odds and that invalid applications return HTTP 422.

## Run locally

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

Open http://127.0.0.1:8000.

## Docker

```powershell
docker build -t credit-risk:local .
docker run --rm -p 8080:8080 credit-risk:local
```

Open http://127.0.0.1:8080. Set `MODEL_DIR` to load the model from another directory.

## Training on AWS (region ap-south-1)

```mermaid
flowchart LR
    CSV[credit_risk_dataset.csv] -->|upload if changed| S3D[(S3 data/raw, versioned)]
    DF[Dockerfile.train] -->|build + push, tag = git commit| ECR[(ECR credit-risk-train)]
    S3D --> JOB[SageMaker Processing job<br/>python -m credit_risk.train]
    ECR --> JOB
    JOB -->|model files| S3M[(S3 models/job-name)]
    JOB -->|params, metrics, plots, model| MLF[SageMaker serverless MLflow]
    MLF -->|auto-sync| REG[SageMaker Model Registry]
```

One-time setup, with AWS credentials for the `dev-aws` profile:

```powershell
$env:AWS_PROFILE = "dev-aws"
cd infra
npx aws-cdk@2 bootstrap      # once per account and region
npx aws-cdk@2 deploy         # S3, ECR, IAM roles, MLflow App
cd ..
```

Train:

```powershell
.\.venv\Scripts\python.exe scripts\launch_training.py --profile dev-aws
```

This uploads the dataset if it changed, builds and pushes the training image,
and runs a SageMaker Processing job on `ml.t3.large` (about 7 minutes, capped at
1 hour by `--max-runtime`). The job logs to the `credit-risk` MLflow experiment
and registers a new version of `credit-risk-xgboost`, which MLflow copies to
the SageMaker Model Registry.

A Processing job is used because new accounts have a quota of 0 for every
training-job instance type. After a quota increase for `ml.m5.large for training
job usage`, run with `--job-type training` to use a Training job instead.

Open the MLflow UI from the SageMaker AI console: **SageMaker AI Studio →
Applications → MLflow → credit-risk-mlflow**.

Remove everything (bucket contents, images and MLflow data included):

```powershell
cd infra; npx aws-cdk@2 destroy --all
```

## Serving on AWS Lambda

```mermaid
flowchart LR
    MLF[MLflow registry<br/>alias champion] -->|fetch_model.py| BM[build/model]
    BM --> IMG[Dockerfile + Lambda Web Adapter]
    IMG -->|cdk deploy CreditRiskServing| L[Lambda credit-risk-api]
    L --> URL[Public Function URL<br/>web form + /predict]
    L --> CW[CloudWatch Logs<br/>one JSON line per prediction]
    GH[GitHub Actions<br/>OIDC, no stored keys] -->|deploy_serving.py| IMG
```

The same `Dockerfile` runs locally and on Lambda. The Lambda Web Adapter
forwards Lambda requests to uvicorn. The model files are baked into the image,
so each deployed image always serves one known model version; `/health`
reports which one.

Promote a model, then deploy it:

```powershell
.\.venv\Scripts\python.exe scripts\promote_model.py --version 1 --profile dev-aws
.\.venv\Scripts\python.exe scripts\deploy_serving.py --profile dev-aws
```

`promote_model.py` sets the MLflow alias `champion` and approves the matching
SageMaker Model Registry package. `deploy_serving.py` downloads the champion,
runs `cdk deploy CreditRiskServing`, and smoke-tests `/health` and `/predict`
on the live URL. It fails if Lambda doesn't serve the expected version.

Lambda settings: 2048 MB memory, 30 s timeout, at most 5 concurrent requests
(caps the cost of the public URL). A cold start takes about 20 seconds;
warm requests take about 150 ms.

### GitHub Actions

| Workflow | Runs on | What it does |
|---|---|---|
| `ci.yml` | every push and pull request | runs the tests and `cdk synth` |
| `deploy.yml` | push to `main`, or by hand | runs CI, then `deploy_serving.py`. When run by hand you can pick a model version |

The deploy job logs in to AWS through OIDC as the role `credit-risk-github-deploy`.
Only the `main` branch of `Vaibhav0PS/FairLend` can assume it. The role can
assume the CDK deploy roles and read the model from MLflow, nothing else.

## Monitoring and retraining

```mermaid
flowchart LR
    API[Lambda API] -->|inputs + probability, one JSON line per request| LOGS[CloudWatch Logs]
    LOGS --> DRIFT[Drift-check Lambda<br/>daily, PSI per feature]
    DRIFT -->|metrics| CW[CloudWatch dashboard credit-risk]
    DRIFT -->|report + predictions| S3[(S3 monitoring/)]
    CW -->|MaxFeaturePSI > 0.25| ALARM[Alarm credit-risk-drift]
    ALARM -->|email| SNS[SNS credit-risk-alerts]
    ALARM -->|EventBridge| PIPE
    subgraph PIPE[SageMaker Pipeline credit-risk-retraining]
        T[Train] --> E[Evaluate vs champion] --> C{better?}
        C -->|yes| R[Register as challenger<br/>PendingManualApproval]
    end
    R -->|approval email| SNS
    C -->|no| N[“not better” email]
    R -.->|you: promote_model.py + Deploy API| API
```

**Drift check.** Each training run saves `reference_profile.json`: decile bins
(numeric inputs) or category shares (categorical inputs) of the training data, and
of the predicted probability. The `credit-risk-drift-check` Lambda runs daily, bins
the last 24 hours of API requests the same way, and computes the population
stability index (PSI) per input. Below 0.1 is stable, above 0.25 is a major shift.
It needs at least 50 requests. The profile is packaged with the Lambda at deploy
time, so it always matches the model being served.

**Retraining with approval.** The drift alarm starts the pipeline. It trains a
new model, scores it and the champion on the same test rows, and registers the
new one only if its PR-AUC is higher. A registered model gets the MLflow alias
`challenger` and the SageMaker approval status `PendingManualApproval`, and you get
an email with both models' metrics. Nothing is deployed until you approve:

```powershell
.\.venv\Scripts\python.exe scripts\promote_model.py --version <N> --profile dev-aws
# then GitHub -> Actions -> Deploy API -> Run workflow
```

Note: the pipeline retrains on the dataset in S3. The live API never learns whether
an applicant defaulted, so a real system would add newly labelled loans to that
dataset before retraining. Until then, retraining on unchanged data produces the
same model, which the pipeline correctly declines to register.

**Demo commands**

```powershell
python scripts/subscribe_alerts.py --email you@example.com --profile dev-aws   # once
python scripts/simulate_traffic.py --n 120 --check --profile dev-aws           # normal traffic
python scripts/simulate_traffic.py --n 250 --drift --check --profile dev-aws   # drift -> alarm -> pipeline
python scripts/start_retraining.py --profile dev-aws                           # start the pipeline by hand
python scripts/start_retraining.py --profile dev-aws --min-gain -1             # force the approval path
```

## API

`POST /predict`

```json
{
  "person_age": 30,
  "person_income": 60000,
  "person_home_ownership": "RENT",
  "person_emp_length": 5,
  "loan_intent": "PERSONAL",
  "loan_grade": "A",
  "loan_amnt": 10000,
  "loan_int_rate": 7.5,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 6
}
```

`loan_percent_income` is computed on the server as `loan_amnt / person_income`.
The response includes the calibrated default probability, the risk band, and
one SHAP factor per input column. `base_value` plus the sum of all
`shap_value`s equals `model_log_odds`.
