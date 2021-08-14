import json

def load_hparams():
    with open('hparams.json') as file:
        data = json.load(file)
        return data
