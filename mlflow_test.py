import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("first_run")

with mlflow.start_run():
    mlflow.log_metric("foo", 15)
    mlflow.log_metric("bar", 20)
    mlflow.log_param("a",20)
