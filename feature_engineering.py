from sklearn.decomposition import PCA
import joblib


class FeatureManipulator:
    def __init__(self, train, test):
        self.train = train
        self.test = test

    def extraction(self):
        self.pca = PCA(n_components=10)
        train = self.pca.fit_transform(self.train)
        test = self.pca.transform(self.test)

        return train, test

    def save(self):
        joblib.dump(self.pca, "pca.joblib")
