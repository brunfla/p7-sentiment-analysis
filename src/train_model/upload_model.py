import mlflow
import mlflow.pyfunc
import os
import json

# Configurer le tracking URI
mlflow.set_tracking_uri("http://mlflow.local")

# Charger les informations du run
mlflow_id_path = "data/output/trained_models/distilbert-base-uncased/mlflow_id.json"
with open(mlflow_id_path, "r") as f:
    mlflow_id = json.load(f)
run_id = mlflow_id["run_id"]

# Enregistrer le modèle
model_dir = "data/output/trained_models/distilbert-base-uncased/final_model"

class CustomModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import ktrain
        self.predictor = ktrain.load_predictor(context.artifacts["model_dir"])

    def predict(self, context, inputs):
        return self.predictor.predict(inputs)

# Enregistrement dans MLflow
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=CustomModel(),
        artifacts={"model_dir": model_dir},
        registered_model_name="distilbert"
    )

print(f"Modèle enregistré avec succès dans le run ID : {run_id}")
