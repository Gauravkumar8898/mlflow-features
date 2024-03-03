from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

path = Path(__file__).parents[1]
data = path / 'data'
model_name="distilbert-base-cased"
hub_model_id_path = "akshatmehta98/distilbert-imdb-mlflow"
output_directory="./output"


