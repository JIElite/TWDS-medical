class ProbabilisticBinaryClassifier:
    def __init__(self, model, prob_threshold=None):
        self.model = model
        self.prob_threshold = prob_threshold

    def __call__(self, X):
        return self.predict(X)

    def predict(self, X):
        if self.prob_threshold:
            return (self.model.predict_proba(X) > self.prob_threshold)[:, 0].astype(int)
        else:
            return self.model.predict_proba(X)
