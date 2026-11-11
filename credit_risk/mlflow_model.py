"""MLflow pyfunc wrapper so the model can be registered and loaded from MLflow."""

import mlflow.pyfunc


class CreditRiskPyfunc(mlflow.pyfunc.PythonModel):
    """Returns the calibrated default probability for each row."""

    def load_context(self, context):
        from credit_risk.model import CreditRiskModel

        self.model = CreditRiskModel.load(context.artifacts["model_dir"])

    def predict(self, context, model_input, params=None):
        return self.model.predict_proba(model_input)
