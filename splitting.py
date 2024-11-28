from sklearn.model_selection import train_test_split


class Splitter:
    def __init__(self, data):
        self.data = data

    def split(self, X, y, random_state=42, train_size=0.8):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=random_state, train_size=train_size
        )

        return X_train, X_test, y_train, y_test

    def target_split(self, target):
        X = self.data.drop([target], axis=1)
        y = self.data[target]

        return X, y
