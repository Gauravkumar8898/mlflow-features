epochs = 1
batch_size = 32
validation_split = 0.2
verbose = 1

artifact_location = "mlflow_artifact"

# Provide an Experiment description that will appear in the UI
experiment_description = (
    "This is the Mnist project. "
    "This experiment contains the produce models for Mnist"
)

# Provide searchable tags that define characteristics of the Runs that
# will be in this Experiment
experiment_tags = {
    "project_name": "Mnist",
    "team": "ml-interns",
    "project_quarter": "Q1-2024",
    "mlflow.note.content": experiment_description,
}
