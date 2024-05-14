# Generic parent class for models, to train and predict
class Model:
    models = {} # name: model
    args = {}

    def __init__(self, **kwargs):
        self.args = kwargs

    def add_model(self, model, name=None):
        self.models[name] = model

    def train_model(self, X, y, name=None):
        print("Parent class training")
        self.models[name].fit(X, y)

    def predict(self, X, name=None):
        return self.models[name].predict_proba(X)