"""The behaviour model: average of a random forest and a gradient-boosting model.
The forest gives the more honest confidence (used for the unfamiliarity score);
gradient boosting is more accurate on its own. Averaging keeps both strengths."""
import numpy as np


class AvgModel:
    def __init__(self, models):
        self.models = models
        self.classes_ = models[0].classes_
        for m in models: assert list(m.classes_) == list(self.classes_)

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)

    def predict(self, X):
        return np.asarray(self.classes_)[self.predict_proba(X).argmax(1)]
