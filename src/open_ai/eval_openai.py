import mlflow
import openai
import os
import pandas as pd
from getpass import getpass

eval_data = pd.DataFrame(
    {
        "inputs": [
            "What is MLflow?",
            "What is Spark?",
        ],
        "ground_truth": [
            "MLflow is an open-source platform for managing the end-to-end machine learning (ML) "
            "lifecycle. It was developed by Databricks, a company that specializes in big data and "
            "machine learning solutions. MLflow is designed to address the challenges that data "
            "scientists and machine learning engineers face when developing, training, and deploying "
            "machine learning models.",
            "Apache Spark is an open-source, distributed computing system designed for big data "
            "processing and analytics. It was developed in response to limitations of the Hadoop "
            "MapReduce computing model, offering improvements in speed and ease of use. Spark "
            "provides libraries for various tasks such as data ingestion, processing, and analysis "
            "through its components like Spark SQL for structured data, Spark Streaming for "
            "real-time data processing, and MLlib for machine learning tasks",
        ],
    }
)

with mlflow.start_run() as run:
    # Logging parameters
    mlflow.log_param("model_name", "gpt-3.5-turbo")
    mlflow.log_param("task", "chat.completions")
    mlflow.log_param("system_prompt", "Answer the following question in two sentences")

    # Wrap "gpt-3.5-turbo" as an MLflow model.
    logged_model_info = mlflow.openai.log_model(
        model="gpt-3.5-turbo",
        task=openai.chat.completions,
        artifact_path="model",
        messages=[
            {"role": "system", "content": "Answer the following question in two sentences"},
            {"role": "user", "content": "{question}"},
        ],
    )

    # Use predefined question-answering metrics to evaluate our model.
    results = mlflow.evaluate(
        logged_model_info.model_uri,
        eval_data,
        targets="ground_truth",
        model_type="question-answering",
        extra_metrics=[mlflow.metrics.toxicity(), mlflow.metrics.latency()],
    )
    print(f"See aggregated evaluation results below: \n{results.metrics}")

    # Logging evaluation metrics
    mlflow.log_metric("latency_mean", results.metrics['latency/mean'])
    mlflow.log_metric("latency_variance", results.metrics['latency/variance'])
    mlflow.log_metric("latency_p90", results.metrics['latency/p90'])
    mlflow.log_metric("exact_match_v1", results.metrics['exact_match/v1'])

    # Evaluation result for each data record is available in `results.tables`.
    eval_table = results.tables["eval_results_table"]
    eval_table.to_html("test.html")
    print(f"See evaluation table below: \n{eval_table}")

