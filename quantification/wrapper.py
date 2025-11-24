import torch
import numpy as np
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin

class WrapperClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model, data, device='cpu'):
        self.model = model
        self.data = data
        self.device = device
        self.classes = np.unique(data.y.cpu().numpy())

        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)

        self._precomputed_probs = None
        self._precompute()

    def _precompute(self):
        self.model.eval()
        with torch.no_grad():
            try:
                logits = self.model(self.data.x, self.data.edge_index)
            except TypeError:
                logits = self.model(self.data.x)

            self._precomputed_probs = F.softmax(logits, dim=1).cpu().numpy()

    def fit(self, X, y=None):
        # No fitting necessary as the classifier is pre-trained
        if self._precomputed_probs is None:
            self._precompute()
        return self

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        indices = np.array(X).flatten()
        return self._precomputed_probs[indices]