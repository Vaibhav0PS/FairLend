# MLOps project progress

Last updated: 2026-11-12. AWS account 707408838833, region ap-south-1, CLI profile `dev-aws`.

## Done

- **Phase 1 (local):** training moved from the notebook to `credit_risk/`, with the
  notebook's F1, threshold and SHAP bugs fixed. The API validates its input.
  Added a Dockerfile and tests (19 passing).
- **Phase 2 (AWS):** the CDK stack `CreditRiskMlops` creates the S3 bucket, ECR repo
  `credit-risk-train`, IAM roles and the MLflow App `credit-risk-mlflow`.
  `scripts/launch_training.py` runs training as a SageMaker Processing job.
  The first run (`credit-risk-train-20261111190859`) registered `credit-risk-xgboost` v1:
  test ROC-AUC 0.952, PR-AUC 0.911.

## Open items

- Changes are not committed yet, so image tags end in `-dirty-<timestamp>`.
- Training-job quota is 0. Optionally request `ml.m5.large for training job usage`
  (code `L-611FA074`) = 1, then use `--job-type training`.
- `credit_risk_model.pkl` and `best_threshold.pkl` are no longer used and can be deleted.
- Choose the decision threshold: the current one maximises F1. A cost-based one would
  catch more defaulters.

## Next

- **Phase 3:** serve the registered model from Lambda (container image + Function URL),
  pushed to ECR. GitHub Actions runs tests, builds and deploys, using OIDC (no stored keys).
- **Phase 4:** SageMaker Pipeline that only promotes a model if it beats the current one.
  Prediction logging, a scheduled drift check, and a CloudWatch alarm that triggers retraining.

## Resuming

```powershell
aws sso login --profile dev-aws      # SSO sessions expire after a few hours
.\.venv\Scripts\python.exe -m pytest # sanity check
```

Nothing bills while idle except a few MB of S3 and about 250 MB of ECR storage.
To remove everything: `cd infra; npx aws-cdk@2 destroy`.
