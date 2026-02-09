import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    """
    Implémentation du perceptron de Rosenblatt (classification binaire).
    """

    def __init__(self, learning_rate=0.01, n_epochs=100):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.W = None
        self.b = None
        self.errors_ = []

    def _activation(self, z):
        """Fonction seuil (Heaviside)."""
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.W = np.zeros(n_features)
        self.b = 0.0
        self.errors_ = []

        for _ in range(self.n_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                z = np.dot(xi, self.W) + self.b
                y_hat = self._activation(z)
                update = self.learning_rate * (yi - y_hat)

                if update != 0:
                    self.W += update * xi
                    self.b += update
                    errors += 1

            self.errors_.append(errors)

        return self

    def predict(self, X):
        z = np.dot(X, self.W) + self.b
        return self._activation(z)

    def plot(self, X, y, resolution=200):
        """
        Visualise les données et la frontière de décision (uniquement en 2D).
        """
        if X.shape[1] != 2:
            raise ValueError("La visualisation n'est possible que pour des données 2D.")

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
        )

        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(grid).reshape(xx.shape)

        plt.figure(figsize=(6, 6))
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Frontière de décision du Perceptron")
        plt.show()
