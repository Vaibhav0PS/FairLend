# MLOps project progress

Last updated: 2026-11-18. AWS account 707408838833, region ap-south-1, CLI profile `dev-aws`.

## Done

- **Phase 1 (local):** training moved from the notebook to `credit_risk/`, with the
  notebook's F1, threshold and SHAP bugs fixed. The API validates its input.
- **Phase 2 (AWS training):** stack `CreditRiskMlops`: S3, ECR, IAM roles, MLflow App
  `credit-risk-mlflow`. Run `credit-risk-train-20261111190859` registered
  `credit-risk-xgboost` v1 (test ROC-AUC 0.952, PR-AUC 0.911).
- **Phase 3 (serving + CI/CD):** v1 is `champion`, served by Lambda `credit-risk-api`
  (stack `CreditRiskServing`) at
  https://4wwn4pzon72bodvf225m7hosb40easaf.lambda-url.ap-south-1.on.aws/
  GitHub Actions deploys through OIDC (role `credit-risk-github-deploy`). GitHub's OIDC
  subject includes repo IDs: `repo:Vaibhav0PS@184383907/FairLend@1403473338:ref:refs/heads/main`.
- **Phase 4 (monitoring + retraining):** daily drift-check Lambda, alarms, dashboard
  `credit-risk`, SNS topic `credit-risk-alerts` (email confirmed), SageMaker Pipeline
  `credit-risk-retraining` with a manual approval gate. 24 tests pass.
  Demo on 2026-11-18:
  - normal traffic: max PSI 0.15 (no alarm)
  - drifted traffic: max PSI 0.62 (loan_int_rate), alarm fired, pipeline started
    automatically 40 s later; new model equal to champion (gain 0.0), not registered
  - forced run (`--min-gain -1`): registered **v2 as challenger, PendingManualApproval**.
    v2 is identical to v1 (same data). Approving it is optional; v1 keeps serving.

- **Docs:** `PROJECT_STORY.md` tells the whole project as a story for the review.
  `scripts/open_mlflow.py` opens the MLflow UI (there is no SageMaker Studio domain).
- **End of 2026-11-18:** nothing running; drift alarm reset to OK so a demo during
  the review triggers a fresh ALARM and pipeline run. Expected cost until the
  review (about 10 days): under $1.

## Open items

- Commit and push the Phase 4 changes (CI will redeploy the API with v1).
- Review demo: `simulate_traffic.py --n 250 --drift --check`, then watch the alarm,
  the email and the pipeline. The pipeline only starts when the alarm changes to
  ALARM, so it must be OK before the demo. Check CloudWatch -> Alarms ->
  credit-risk-drift; if it still shows ALARM, reset it with
  `aws cloudwatch set-alarm-state --alarm-name credit-risk-drift --state-value OK --state-reason reset --profile dev-aws --region ap-south-1`.
- After grading: `cd infra; npx aws-cdk@2 destroy --all`.
- Training-job quota is 0; Processing jobs are used instead (works fine).
- `render.yaml`, `credit_risk_model.pkl` and `best_threshold.pkl` are no longer needed.
- Decision threshold maximises F1; a cost-based one would catch more defaulters.
- Prediction logs hold applicant inputs (2-week retention, S3 copies under monitoring/).
  Fine for a course demo with a public dataset; real applicant data would need a
  privacy review.

## Resuming

```powershell
aws sso login --profile dev-aws      # SSO sessions expire after a few hours
# start Docker Desktop before deploying
.\.venv\Scripts\python.exe -m pytest # sanity check
```

To remove everything: `cd infra; npx aws-cdk@2 destroy --all`.
