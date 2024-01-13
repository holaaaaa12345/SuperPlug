"""
This is my custom and purely python backend for the SuperPlug. 
The OOP structure resembles that of Sklearn. In fact, I try to 
mimick Sklearn but without all the fancy code and exception handling.
Most comments and docs are written by chatgpt.
"""


import numpy as np
from metrics import *
from abc import ABC, abstractmethod


class BaseEstimator:
    """
    Base class for all estimators.

    Methods:
        - set_params(**params): Set parameters on the estimator.

    """

    def set_params(self, **params):
        """
        Set parameters on the estimator.

        Parameters:
            - **params: Keyword arguments representing parameters to be set.

        Returns:
            - self: Returns the instance itself.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_params(self):
        """Get parameters for the estimator."""
        params = {}
        for key in self.__dict__:
            if not (key.startswith("_") or key.startswith("__") or key.endswith("_")):
                value = getattr(self, key)
                params[key] = value
        return params

class TransformerMixin:
    """
    Mixin class for transformers.

    Methods:
        - fit_transform(X, **fit_params): Fit the transformer on the input data and transform it.

    """

    def fit_transform(self, X, **fit_params):
        """
        Fit the transformer on the input data and transform it.

        Parameters:
            - X: Input data.
            - **fit_params: Additional keyword arguments for fitting.

        Returns:
            - Transformed data.
        """
        return self.fit(X, **fit_params).transform(X)


class KFold:
    """
    K-Folds cross-validator.

    Parameters:
        - n_splits (int, optional): Number of folds. Default is 5.
        - shuffle (bool, optional): Whether to shuffle the data before splitting. Default is True.
        - random_seed (int or None, optional): Seed used by the random number generator. If None, the random
          number generator is initialized based on the system time. Default is None.

    Methods:
        - split(X, y=None): Generate indices to split data into training and test sets.

    Example:
        ```
        kfold = KFold(n_splits=5, shuffle=True, random_seed=42)
        for train_indices, test_indices in kfold.split(X):
            # Use train_indices and test_indices for training and testing
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
        ```
    """

    def __init__(self, n_splits=5, shuffle=True, random_seed=None):
        """
        Initialize KFold instance.

        Parameters:
            - n_splits (int, optional): Number of folds. Default is 5.
            - shuffle (bool, optional): Whether to shuffle the data before splitting. Default is True.
            - random_seed (int or None, optional): Seed used by the random number generator. If None, the random
              number generator is initialized based on the system time. Default is None.
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.indices = None

    def split(self, X, y=None):
        """
        Generate indices to split data into training and test sets.

        Parameters:
            - X (array-like): The data to split.
            - y (array-like or None, optional): The target variable for supervised learning. Default is None.

        Yields:
            - train_indices, test_indices: The indices to split the data into training and test sets.
        """
        np.random.seed(self.random_seed)
        n_samples = len(X)
        self.indices = np.arange(n_samples)

        if self.shuffle:
            np.random.shuffle(self.indices)

        fold_size = n_samples // self.n_splits
        remainder = n_samples % self.n_splits

        folds = []
        start = 0

        for i in range(self.n_splits):
            end = start + fold_size + (1 if i < remainder else 0)
            test_indices = self.indices[start:end]
            train_indices = np.concatenate([self.indices[:start], self.indices[end:]])

            yield train_indices, test_indices
            start = end

class RandomizedSearchCV():
    """
    Randomized Search Cross-Validator.

    Parameters:
        - estimator: The model to be optimized.
        - param_distributions (dict): Dictionary with parameters names (string) as keys and distributions
          or lists of parameters to try as values.
        - metric (callable, optional): The metric function to be maximized. Default is r2_score.
        - n_iter (int, optional): Number of parameter settings that are sampled. Default is 20.

    Methods:
        - fit_evaluate_fold(X, y): Fit and evaluate the model on each fold.
        - get_parameter_permutation(param_distributions): Generate unique permutations of hyperparameters.
        - fit(X, y): Run random search to find the best hyperparameters and fit the model.

    Example:
        ```
        # Example usage with a hypothetical model and dataset
        model = MyModel()
        param_dist = {'param1': [1, 2, 3], 'param2': [0.1, 0.2, 0.3]}
        random_search = RandomizedSearchCV(model, param_dist, metric=my_custom_metric, n_iter=10)
        random_search.fit(X_train, y_train)
        print("Best hyperparameters:", random_search.best_params)
        ```
    TODO:
        passing metric is still under constrution
    """

    def __init__(self, estimator, param_distributions, metric=r2_score, n_iter=20):
        """
        Initialize RandomizedSearchCV instance.

        Parameters:
            - estimator: The model to be optimized.
            - param_distributions (dict): Dictionary with parameters names (string) as keys and distributions
              or lists of parameters to try as values.
            - metric (callable, optional): The metric function to be maximized. Default is r2_score.
            - n_iter (int, optional): Number of parameter settings that are sampled. Default is 20.
        """
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.metric = metric
    
    def fit_evaluate_fold(self, X, y):
        """
        Fit and evaluate the model on each fold.

        Parameters:
            - X: The data to fit.
            - y: The target variable.

        Returns:
            - metric_train: Array of training metrics for each fold.
            - metric_test: Array of testing metrics for each fold.
        """
        metric_train = []
        metric_test = []
        fold = KFold(shuffle=True)
        
        for train_idx, test_idx in fold.split(X, y):
            X_train_cv = X[train_idx]
            y_train_cv = y[train_idx]
            X_test_cv = X[test_idx]
            y_test_cv = y[test_idx] 
            self.estimator.fit(X_train_cv, y_train_cv)            
            
            train_eval = self.estimator.score(X_train_cv, y_train_cv)
            test_eval = self.estimator.score(X_test_cv, y_test_cv)
            metric_train.append(train_eval)
            metric_test.append(test_eval)
            
        return np.array(metric_train), np.array(metric_test)
    
    def get_parameter_permutation(self, param_distributions):
        """
        Generate unique permutations of hyperparameters.

        Parameters:
            - param_distributions (dict): Dictionary with parameters names (string) as keys and distributions
              or lists of parameters to try as values.

        Returns:
            - unique_permutation: List of unique hyperparameter permutations.
        """
        raw_permutation = []
        unique_permutation = []
        while len(unique_permutation) < self.n_iter:
            sampled_hyperparameters = {param: np.random.choice(values) 
                                       for param, values in param_distributions.items()}
            raw_permutation.append(sampled_hyperparameters)

            # Check if the exact parameter permutation already existed
            current_unique_len = len(set([tuple(i.items()) for i in raw_permutation]))
            last_unique_len = len(unique_permutation)
            if current_unique_len > last_unique_len:
                unique_permutation.append(sampled_hyperparameters)
                
        return unique_permutation
            
    def fit(self, X, y):
        """
        Run random search to find the best hyperparameters and refit the model.

        Parameters:
            - X: The data to fit.
            - y: The target variable.

        Returns:
            - self: Returns the instance itself.
        """
        combination = self.get_parameter_permutation(self.param_distributions)
        result_dict = {"hyperparameters":[], "train_mean": [], "test_mean":[]}
        for i in combination:
            self.estimator.set_params(**i)

            train_score, test_score = self.fit_evaluate_fold(X, y)
            result_dict["hyperparameters"].append(i)
            result_dict["train_mean"].append(train_score.mean())
            result_dict["test_mean"].append(test_score.mean())
        
        best_idx = np.argmax(result_dict["test_mean"])
        self.best_params = combination[best_idx]
        self.estimator.set_params(**self.best_params)
        self.best_estimator_ = self.estimator.fit(X, y)

        return self


class Pipeline:
    """
    Simple implementation of a data processing pipeline.

    Parameters:
        - steps (list): List of tuples where each tuple contains a name and a data processing step.

    Methods:
        - fit(X, y=None): Fit the pipeline on the input data.
        - transform(X): Transform the input data using the pipeline.
    """

    def __init__(self, steps):
        """
        Initialize Pipeline instance.

        Parameters:
            - steps (list): List of tuples where each tuple contains a name and a data processing step.
        """
        self.steps = steps
        
    def fit(self, X, y=None):
        """
        Fit the pipeline on the input data.

        Parameters:
            - X: Input data.
            - y (optional): Target variable.

        Returns:
            - self: Returns the instance itself.
        """
        Xt = X
        for name, step in self.steps:
            Xt = step.fit_transform(Xt)
        return self
               
    def transform(self, X):
        """
        Transform the input data using the pipeline.

        Parameters:
            - X: Input data.

        Returns:
            - Xt: Transformed data.
        """
        Xt = X
        for name, step in self.steps:
            Xt = step.transform(Xt)
        return Xt


class StandardScaler(TransformerMixin):
    """
    StandardScaler scales input features by removing the mean and scaling to unit variance.

    Methods:
        - fit(X): Compute the mean and standard deviation of the input data.
        - transform(X): Scale the input data using the computed mean and standard deviation.
        - fit_transform(X): Fit the scaler and transform the input data in a single step.
    """

    def __init__(self):
        """
        Initialize StandardScaler instance.
        """
        self.mean = None
        self.std = None
        
    def fit(self, X):
        """
        Compute the mean and standard deviation of the input data.

        Parameters:
            - X: Input data.

        Returns:
            - self: Returns the instance itself.
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self
    
    def transform(self, X):
        """
        Scale the input data using the computed mean and standard deviation.

        Parameters:
            - X: Input data.

        Returns:
            - X_scaled: Scaled data.
        """
        X_scaled = (X - self.mean) / self.std
        return X_scaled
    
    def fit_transform(self, X):
        """
        Fit the scaler and transform the input data in a single step.

        Parameters:
            - X: Input data.

        Returns:
            - X_scaled: Scaled data.
        """
        self.fit(X) 
        X_scaled = self.transform(X)
        return X_scaled

class SimpleImputer(TransformerMixin):
    """
    Simple imputer for handling missing values in a dataset.

    Parameters:
        - strategy (str, optional): The imputation strategy. 'mean' for mean imputation, 'mode' for mode imputation.
        - **kwargs: Additional keyword arguments.

    Methods:
        - fit(X): Compute imputation statistics from the input data.
        - transform(X): Impute missing values in the input data.
    """

    def __init__(self, strategy='mean', **kwargs):
        """
        Initialize SimpleImputer instance.

        Parameters:
            - strategy (str, optional): The imputation strategy. 'mean' for mean imputation, 'mode' for mode imputation.
            - **kwargs: Additional keyword arguments.
        """
        self.strategy = strategy
        
    def fit(self, X):
        """
        Compute imputation statistics from the input data.

        Parameters:
            - X: Input data.

        Returns:
            - self: Returns the instance itself.
        """
        self.statistics_ = []
        for i in range(X.shape[1]):
            column = X[:, i]
            if self.strategy == "mean":
                self.statistics_.append(column[column == column].mean()) 
            else:
                mode, counts = np.unique(column[column == column], return_counts=True)  
                index = np.argmax(counts)
                self.statistics_.append(mode[index])
                
    def transform(self, X):
        """
        Impute missing values in the input data.

        Parameters:
            - X: Input data.

        Returns:
            - X_imputed: Input data with missing values imputed.
        """
        X_imputed = np.copy(X)
        for i in range(X.shape[1]):
            if self.strategy == "mean":
                missing_bool = (X_imputed[:, i] != X_imputed[:, i])
            else:
                missing_bool = (X_imputed[:, i] == "")
            X_imputed[missing_bool, i] = self.statistics_[i]

        return X_imputed


class OneHotEncoder(TransformerMixin):
    """
    Simple one-hot encoder for categorical variables.

    Parameters:
        - drop (str, optional): Specifies a category to drop during one-hot encoding. Default is None.
        - **kwargs: Additional keyword arguments.

    Methods:
        - fit(X): Compute categories for one-hot encoding.
        - transform(X): One-hot encode the input data.
    """

    def __init__(self, drop=None, **kwargs):
        """
        Initialize OneHotEncoder instance.

        Parameters:
            - drop (str, optional): Specifies a category to drop during one-hot encoding. Default is None.
            - **kwargs: Additional keyword arguments.
        """
        self.drop = drop

    def fit(self, X):
        """
        Compute categories for one-hot encoding.

        Parameters:
            - X: Input data.

        Returns:
            - self: Returns the instance itself.
        """
        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]

    def transform(self, X):
        """
        One-hot encode the input data.

        Parameters:
            - X: Input data.

        Returns:
            - X_encoded: One-hot encoded data.
        """
        num_samples = X.shape[0]
        
        # Calculate total output size
        num_features = sum(len(cats) for cats in self.categories_)  
        if self.drop == "first":
            num_features -= len(self.categories_)
            
        X_encoded = np.zeros((num_samples, num_features)) 
        column_counter = 0
        
        for i, cats in enumerate(self.categories_):
            n_categories = len(cats)
            if self.drop == "first":
                # Leave out first category
                n_categories -= 1
                cats = cats[1:]
                
            for j, cat in enumerate(cats): 
                encoded_column = (X[:, i] == cat).astype(int)
                X_encoded[:, column_counter] = encoded_column  
                column_counter += 1
        
        return X_encoded

class BaseRegression(BaseEstimator):
    """
    Base class for regression models.

    Methods:
        - score(X, y, metric=r2_score): Compute the model's performance score on the given data.

    Note:
        All regression models should inherit from this class.
    """

    def score(self, X, y, metric=r2_score):
        """
        Compute the model's performance score on the given data.

        Parameters:
            - X: Input data.
            - y: Target variable.
            - metric (callable, optional): The scoring metric function. Default is r2_score.

        Returns:
            - Score value.
        """
        prediction = self.predict(X)
        return metric(y, prediction)


class LinearModel():
    """
    Base class for linear models.

    Note:
        This class is not fully implemented and serves as a base for other linear models.
    """

    def lin_combination(self, X):
        """
        Compute the linear combination of features.

        Parameters:
            - X: Input features.

        Returns:
            - Linear combination of features.
        """
        return X @ self.parameters_


class LinearRegression(LinearModel, BaseRegression):
    """
    Linear regression model.

    Methods:
        - fit(X, y): Fit the linear regression model.
        - predict(X): Make predictions using the fitted model.

    Attributes:
        - coef_: Coefficients of the linear regression model.
        - intercept_: Intercept term of the linear regression model.
    """

    def fit(self, X, y):
        """
        Fit the linear regression model.

        Parameters:
            - X: Input features.
            - y: Target variable.

        Returns:
            - self: Returns the instance itself.
        """
        n = len(X)
        X = np.hstack((np.ones((n, 1)), X))  # Use np.ones for the column of ones
        self.parameters_ = np.linalg.lstsq(X, y, rcond=None)[0]
        self.coef_ = self.parameters_[1:]
        self.intercept_ = self.parameters_[0]
        return self

    def predict(self, X):
        """
        Make predictions using the fitted model.

        Parameters:
            - X: Input features.

        Returns:
            - prediction: Predicted values.
        """
        X = np.hstack((np.array([1] * len(X)).reshape(-1, 1), X))
        prediction = super().lin_combination(X)
        return prediction

class Ridge(LinearModel, BaseRegression):
    """
    Ridge regression model implementation with NumPy. Notice the difference between
    this and sklearn's implementation. This implementation applies bias all the way
    to the intercept, whereas that of sklearn does not.

    Attributes:
        alpha: The L2 regularization parameter.
        coef_: The estimated model coefficients.
        intercept_: The estimated model intercept.
    """

    def __init__(self, alpha=1.0):
        """
        Initialize the Ridge model with the given L2 regularization parameter.

        Args:
            alpha: The L2 regularization parameter (default: 1.0).
        """
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """
        Fit the Ridge model to the training data.

        Args:
            X: The training features (NumPy array of shape (n_samples, n_features)).
            y: The target values (NumPy array of shape (n_samples,)).
        """
        # Add intercept term
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # Precompute terms for efficiency
        n_samples, n_features = X.shape
        XtX = X.T @ X
        Xty = X.T @ y

        # Closed-form solution
        I = np.eye(n_features)
        self.parameters_ = np.linalg.inv(XtX + self.alpha * I) @ Xty
        self.intercept_ = self.parameters_[0]
        self.coef_ = self.parameters_[1:]

        return self

    def predict(self, X):
        """
        Predict target values for new data using the fitted model.

        Args:
            X: The new features (NumPy array of shape (n_samples, n_features)).

        Returns:
            The predicted target values (NumPy array of shape (n_samples,)).
        """

        # Add intercept term
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        prediction = super().lin_combination(X)

        return prediction


class KNNBase(ABC, BaseRegression):
    """
    K-Nearest Neighbors (KNN) base class for regression and classification.

    Parameters:
    - n_neighbors (int): Number of neighbors to consider during prediction.

    Methods:
    - fit(X, y): Fit the KNN model with training data.
    - predict(X): Predict target values for new data points.

    Abstract Method:
    - prediction_measure(nearest_targets): Define the prediction measure.

    Attributes:
    - n_neighbors (int): Number of neighbors.
    - X_ref_ (numpy.ndarray): Training data features.
    - y_ref_ (numpy.ndarray): Training data labels.
    """

    def __init__(self, n_neighbors=5):
        """Initialize KNN with the specified number of neighbors."""
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fit the KNN model with training data."""
        self.X_ref_ = X
        self.y_ref_ = y
        return self

    def predict(self, X):
        """Predict target values for new data points."""
        distances = np.linalg.norm(X[:, np.newaxis] - self.X_ref_, axis=2)
        nearest_indices = np.argsort(distances)[:, :self.n_neighbors]
        nearest_targets = self.y_ref_[nearest_indices]
        predictions = np.apply_along_axis(self.prediction_measure, axis=1, arr=nearest_targets)
        return predictions

    @staticmethod
    def euclidean_distance(x1, x2):
        """Compute the Euclidean distance between two vectors."""
        return np.linalg.norm(x1 - x2)

    @abstractmethod
    def prediction_measure(self, nearest_targets):
        """Abstract method to define the prediction measure."""


class KNeighborsRegressor(KNNBase):
    """
    KNN regressor for predicting continuous values.

    Method:
    - prediction_measure(nearest_targets): Calculate the mean of nearest targets.
    """

    def prediction_measure(self, nearest_targets):
        """Calculate the mean of nearest targets for regression."""
        return np.mean(nearest_targets)


class KNNClassifier(KNNBase):
    """
    KNN classifier for predicting discrete values.

    Method:
    - prediction_measure(nearest_targets): Predict the mode of nearest targets.
    """

    def prediction_measure(self, nearest_targets):
        """Predict the mode of nearest targets for classification."""
        return int(np.argmax(np.bincount(nearest_targets)))