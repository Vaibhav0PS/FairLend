# MLOps project progress

Last updated: 2026-11-12. AWS account 707408838833, region ap-south-1, CLI profile `dev-aws`.

## Done

- **Phase 1 (local):** training moved from the notebook to `credit_risk/`, with the
  notebook's F1, threshold and SHAP bugs fixed. The API validates its input.
  Added a Dockerfile and tests (19 passing).
- **Phase 2 (AWS training):** the CDK stack `CreditRiskMlops` creates the S3 bucket, ECR repo
  `credit-risk-train`, IAM roles and the MLflow App `credit-risk-mlflow`.
  `scripts/launch_training.py` runs training as a SageMaker Processing job.
  Run `credit-risk-train-20261111190859` registered `credit-risk-xgboost` v1:
  test ROC-AUC 0.952, PR-AUC 0.911.
- **Phase 3 (serving + CI/CD):** v1 is promoted to `champion` and served by the Lambda
  function `credit-risk-api` (stack `CreditRiskServing`) at
  https://4wwn4pzon72bodvf225m7hosb40easaf.lambda-url.ap-south-1.on.aws/
  GitHub OIDC provider and deploy role `credit-risk-github-deploy` exist.
  Workflows `ci.yml` and `deploy.yml` are written but not pushed yet.

## Open items

- Push to GitHub to run the workflows for the first time.
- Training-job quota is 0. Optionally request `ml.m5.large for training job usage`
  (code `L-611FA074`) = 1, then use `--job-type training`.
- `render.yaml`, `credit_risk_model.pkl` and `best_threshold.pkl` are no longer needed.
- Choose the decision threshold: the current one maximises F1. A cost-based one would
  catch more defaulters.
- CDK pushes each serving image to the bootstrap ECR repo without a cleanup rule.
  Images are about 170 MB each, so this costs cents, but `cdk gc` can clean it up.

## Next

- **Phase 4:** SageMaker Pipeline that only promotes a model if it beats the champion.
  Prediction logging to S3, a scheduled drift check, and a CloudWatch alarm that
  triggers retraining.

## Resuming

```powershell
aws sso login --profile dev-aws      # SSO sessions expire after a few hours
# start Docker Desktop before deploying
.\.venv\Scripts\python.exe -m pytest # sanity check
```

Nothing bills while idle except a few MB of S3 and well under 1 GB of ECR storage.
To remove everything: `cd infra; npx aws-cdk@2 destroy --all`.
