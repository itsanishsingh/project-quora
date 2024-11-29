from sklearn.ensemble import RandomForestClassifier
import joblib


class Modeller:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def rand_forest_clf(self):
        clf = RandomForestClassifier(
            max_leaf_nodes=20,
            n_jobs=-1,
            random_state=42,
            verbose=True,
        )
        self.model = clf

    def fit(self):
        self.model.fit(self.X, self.y)

    def predict(self, test):
        pred = self.model.predict(test)
        return pred

    def save(self):
        joblib.dump(self.model, "model_w2v.joblib")
