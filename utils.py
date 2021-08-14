import json

def load_hparams():
    with open('hparams.json') as file:
        data = json.load(file)
        print(json.dumps(data, indent=4))
        return data
