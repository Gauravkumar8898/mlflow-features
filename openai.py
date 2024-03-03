from mlflow.deployments import get_deploy_client


def main():
    client = get_deploy_client("http://localhost:5000")

    print(f"OpenAI endpoints: {client.list_endpoints()}\n")
    print(f"OpenAI endpoint info: {client.get_endpoint(endpoint='completions')}\n")

    # Completions example
    # response_completions = client.predict(
    #     endpoint="completions",
    #     inputs={
    #         "prompt": "How many patties could be stacked on a cheeseburger before issues arise?",
    #         "max_tokens": 200,
    #         "temperature": 0.25,
    #     },
    # )
    # print(f"OpenAI completions response: {response_completions}")

    # Chat example
    response_chat = client.predict(
        endpoint="chat",
        inputs={
            "messages": [
                {
                    "role": "user",
                    "content": "Explain me about mlflow",
                }
            ]
        },
    )
    print(f"OpenAI chat response: {response_chat}")

    # Embeddings example
    # response_embeddings = client.predict(
    #     endpoint="embeddings",
    #     inputs={
    #         "input": "When you say 'enriched', what exactly are you enriching the cereal with?"
    #     },
    # )
    # print(f"OpenAI response for embeddings: {response_embeddings}")


if __name__ == "__main__":
    main()