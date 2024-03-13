import mlflow
import openai
import pandas as pd

# Chat
with mlflow.start_run():
    info = mlflow.openai.log_model(
        model="gpt-3.5-turbo",
        task=openai.chat.completions,
        messages=[{"role": "user", "content": "Tell me a joke about {animal}."}],
        artifact_path="model",
    )
    model = mlflow.pyfunc.load_model(info.model_uri)
    df = pd.DataFrame({"animal": ["cats", "dogs"]})
    print(model.predict(df))


# Embeddings
# with mlflow.start_run():
#     info = mlflow.openai.log_model(
#         model="text-embedding-ada-002",
#         task=openai.embeddings,
#         artifact_path="embeddings",
#     )
#     model = mlflow.pyfunc.load_model(info.model_uri)
#     print(model.predict(["hello", "world"]))