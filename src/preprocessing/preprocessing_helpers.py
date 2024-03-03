from datasets import load_dataset,load_metric
import logging
logging.basicConfig(level=logging.INFO)


def fetch_data():  # Adjust the number of rows as needed
    train_data, test_data = load_dataset("imdb", split=['train', 'test'])
    train_data = train_data.shuffle(seed=42)
    train_data = train_data.select([i for i in range(6000)])
    test_data = test_data.shuffle(seed=42)
    test_data = test_data.select([i for i in range(6000)])
    logging.info('Train data:', train_data)
    logging.info('Test data:', test_data)
    return train_data, test_data