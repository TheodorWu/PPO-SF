import json

def load_hparams():
    """Load content of hparams.json (must be at project root) and pretty print to console."""
    with open('hparams.json') as file:
        data = json.load(file)
        print(json.dumps(data, indent=4))
        return data
