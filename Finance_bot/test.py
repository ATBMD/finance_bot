from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer
from rasa_nlu import config

trainer = Trainer(config.load("./training_data/config_spacy.yml"))
# Load the training data
training_data = load_data('./training_data/rasa_dataset_training.json')
# Create an interpreter by training the model
interpreter = trainer.train(training_data)