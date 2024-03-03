import os
import numpy as np
import pandas as pd
from transformers import (AutoModelForSequenceClassification,AutoTokenizer,
    Trainer,
    TrainingArguments)
import mlflow
from datasets import load_dataset,load_metric
import logging
import os
from huggingface_hub import login
from src.utils.constants import model_name,hub_model_id_path,output_directory
class Mlflow_Pipeline_Transformer:
    def __init__(self):
        login()
        logging.basicConfig(level=logging.INFO)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, num_labels=2)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.metric=load_metric("accuracy")
        os.environ["MLFLOW_EXPERIMENT_NAME"] = "trainer-mlflow-demo"
        os.environ["MLFLOW_FLATTEN_PARAMS"] = "1"
    def fetch_data(self, num_rows=6000):  # Adjust the number of rows as needed
        train_data, test_data = load_dataset("imdb", split=['train', 'test'])

        # Shuffle and select a subset of the training and test data
        train_data = train_data.shuffle(seed=42)
        train_data = train_data.select([i for i in range(num_rows)])

        test_data = test_data.shuffle(seed=42)
        test_data = test_data.select([i for i in range(num_rows)])

        logging.info('Train data:', train_data)
        logging.info('Test data:', test_data)

        return train_data, test_data


    def tokenise(self,dataset):
        return self.tokenizer(dataset['text'],padding="max_length",truncation=True)


    def compute_metrics(self,y_pred):
        logits,labels=y_pred
        predictions=np.argmax(logits,axis=-1)
        return self.metric.compute(predictions,references=labels)



    def runner(self):
        train_data,test_data=self.fetch_data()
        train_data=train_data.map(self.tokenise,batched=True)
        test_data=test_data.map(self.tokenise,batched=True)
        training_args = TrainingArguments(
            hub_model_id=hub_model_id_path,
            num_train_epochs=1,
            output_dir=output_directory,
            logging_steps=100,
            save_strategy="epoch",
            push_to_hub=True,
            per_device_train_batch_size=3,
            per_device_eval_batch_size=3
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        mlflow.end_run()
        trainer.push_to_hub()







