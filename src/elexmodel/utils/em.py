import numpy as np
from scipy.stats import multivariate_normal


class GMM:
    def __init__(self, n_components, max_iter=100, tol=1e-6):
        """
        Initialize the Gaussian Mixture Model.

        Parameters:
        n_components : Number of Gaussian components (clusters).
        max_iter     : Maximum number of iterations for the EM algorithm.
        tol          : Tolerance for convergence.
        """
        self.n_components = n_components  # Number of Gaussian components
        self.max_iter = max_iter  # Maximum number of iterations
        self.tol = tol  # Convergence tolerance
        self.means = None  # Means of the Gaussians
        self.covariances = None  # Covariance matrices of the Gaussians
        self.weights = None  # Mixing coefficients (weights)
        self.log_likelihoods = []  # Log-likelihood values during training

    def _expectation_step(self, X):
        """
        Perform the E-step of the EM algorithm, computing the responsibilities.

        Parameters:
        X : Data points (N x D), where N is the number of data points and D is the dimensionality.

        Returns:
        responsibilities : N x k matrix of responsibilities for each component.
        """
        N = X.shape[0]
        responsibilities = np.zeros((N, self.n_components))

        for i in range(self.n_components):
            # For each Gaussian component, calculate the pdf and multiply by the weight
            rv = multivariate_normal(mean=self.means[i], cov=self.covariances[i])
            responsibilities[:, i] = self.weights[i] * rv.pdf(X)

        # Normalize the responsibilities
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def _maximization_step(self, X, responsibilities):
        """
        Perform the M-step of the EM algorithm, updating the parameters.

        Parameters:
        X               : Data points (N x D).
        responsibilities : N x k matrix of responsibilities.

        Updates:
        - Means
        - Covariances
        - Weights (mixing coefficients)
        """
        N, D = X.shape
        weights = responsibilities.mean(axis=0)
        means = np.zeros((self.n_components, D))
        covariances = np.zeros((self.n_components, D, D))

        for i in range(self.n_components):
            resp_sum = responsibilities[:, i].sum()

            # Update mean
            means[i] = np.sum(X * responsibilities[:, i, np.newaxis], axis=0) / resp_sum

            # Update covariance
            diff = X - means[i]
            covariances[i] = np.dot((responsibilities[:, i, np.newaxis] * diff).T, diff) / resp_sum

        self.weights = weights
        self.means = means
        self.covariances = covariances

    def fit(self, X):
        """
        Fit the Gaussian Mixture Model to the data using the EM algorithm.

        Parameters:
        X : Data points (N x D), where N is the number of data points and D is the dimensionality.
        """
        N, D = X.shape

        # Initialize parameters randomly
        self.means = X[np.random.choice(N, self.n_components, replace=False)]
        self.covariances = np.array([np.eye(D)] * self.n_components)
        self.weights = np.ones(self.n_components) / self.n_components

        self.log_likelihoods = []  # Clear previous log-likelihoods

        for iteration in range(self.max_iter):
            # E-step: Compute responsibilities
            responsibilities = self._expectation_step(X)

            # M-step: Update parameters
            self._maximization_step(X, responsibilities)

            # Calculate the log-likelihood for this iteration
            log_likelihood = np.sum(
                np.log(
                    np.sum(
                        [
                            w * multivariate_normal(m, c).pdf(X)
                            for w, m, c in zip(self.weights, self.means, self.covariances)
                        ],
                        axis=0,
                    )
                )
            )
            self.log_likelihoods.append(log_likelihood)

            # Check for convergence
            if iteration > 0 and abs(log_likelihood - self.log_likelihoods[-2]) < self.tol:
                print(f"Converged at iteration {iteration}")
                break

    def predict_proba(self, X):
        """
        Predict the probabilities (responsibilities) of each data point belonging to each component.

        Parameters:
        X : Data points (N x D).

        Returns:
        responsibilities : N x k matrix of responsibilities for each component.
        """
        return self._expectation_step(X)

    def predict(self, X):
        """
        Predict the component (Gaussian) that each data point most likely belongs to.

        Parameters:
        X : Data points (N x D).

        Returns:
        labels : N-dimensional array of predicted labels (component assignments).
        """
        responsibilities = self.predict_proba(X)
        return np.argmax(responsibilities, axis=1)

    def score(self, X):
        """
        Calculate the log-likelihood of the data under the current model.

        Parameters:
        X : Data points (N x D).

        Returns:
        log_likelihood : Log-likelihood of the data.
        """
        log_likelihood = np.sum(
            np.log(
                np.sum(
                    [
                        w * multivariate_normal(m, c).pdf(X)
                        for w, m, c in zip(self.weights, self.means, self.covariances)
                    ],
                    axis=0,
                )
            )
        )
        return log_likelihood


# Example usage:
X = np.random.randn(300, 2)  # Random 2D data

# Create GMM instance with 3 components
gmm = GMM(n_components=3)

# Fit the model
gmm.fit(X)

# Predict component labels
labels = gmm.predict(X)

# Print the means
print("Fitted means:", gmm.means)
