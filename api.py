"""API to connect to either custom backend or to sklearn. This API
   generates model objects, already equipped with the train and test data,
   that can be fit by the client."""


"""Set this variable to be true if you want to use Sklearn as a backend.
   The default is False"""
USE_SKLEARN = False
if USE_SKLEARN:
	from sklearn.preprocessing import StandardScaler, OneHotEncoder
	from sklearn.model_selection import train_test_split
	from sklearn.linear_model import Lasso, Ridge, LinearRegression
	from sklearn.impute import SimpleImputer
	from sklearn.ensemble import RandomForestRegressor
	from sklearn.base import BaseEstimator, TransformerMixin
	from sklearn.tree import DecisionTreeRegressor
	from sklearn.pipeline import Pipeline
	from sklearn.model_selection import RandomizedSearchCV
	from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
else:
	from backend import *

import numpy as np


class FinalAlgo():
	
	def __init__(self, estimator, all_data, **kwargs):

		self._all_data = all_data
		self.estimator_ = estimator

	def fit(self):

		if hasattr(self.estimator_, "_param_space"):
			cv_search = RandomizedSearchCV(estimator=self.estimator_, 
							 param_distributions=self.estimator_._param_space,
							 n_iter=10)
			cv_search.fit(self._all_data["X_train"], self._all_data["y_train"])
			estimator_final = cv_search.best_estimator_
			# print(estimator_final.alpha)

		else:
			estimator_final = self.estimator_.fit(self._all_data["X_train"], self._all_data["y_train"])

		# Surely there's a better way to do this -__-
		setattr(estimator_final, "_all_data", self._all_data)
		return estimator_final



class CustomClassification:

	def get_score(self):
		y_pred_train = self.predict(X_train)

class CustomRegression:

	def evaluate(self):
		all_data = self._all_data
		metrics = {"r2" : r2_score, 
				   "mse" : mean_squared_error, 
				   "mad" : mean_absolute_error}
		dict_score = {}

		for name, metric in metrics.items():
			for part in ["train", "test"]:
				y_pred = self.predict(all_data[f"X_{part}"])
				y_true = all_data[f"y_{part}"]
				dict_score[f"{name}_{part}"] = np.round(metric(y_true, y_pred), 4)

		return dict_score



class LinearRegression(LinearRegression, CustomRegression):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._all_data = None

class Ridge(Ridge, CustomRegression):

	def __init__(self, alpha=1, **kwargs):
		super().__init__(alpha, **kwargs)
		self._all_data = None
		self._param_space = {"alpha": np.linspace(0.0001, 1, 100)}


# class Lasso(Lasso, CustomRegression):

# 	def __init__(self, alpha=1, **kwargs):
# 		super().__init__(alpha, **kwargs)
# 		self.all_data = None
# 		self.param_space = {"alpha": np.linspace(0.001, 1, 100)}

# class DecisionTreeRegressor(DecisionTreeRegressor, CustomRegression):

# 	def __init__(self, max_depth=0, **kwargs):
# 		super().__init__(max_depth=max_depth, **kwargs)
# 		self.all_data = None
# 		self.param_space = {"max_depth": np.arange(1, 31)}


class StructuredArrImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
    	self.imputer_num = SimpleImputer(strategy="mean")
    	self.imputer_cat = SimpleImputer(strategy="most_frequent", missing_values="")
    	# self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', 
    	# 									 drop="first")
    	# self.standard_scaler = StandardScaler()

        
    def classify_arrays(self, arrays):

	    numeric_arrays = []
	    non_numeric_arrays = []

	    for arr in arrays:
	        if np.issubdtype(arr.dtype, np.number):
	            numeric_arrays.append(arr.reshape(-1, 1))
	        else:
	            non_numeric_arrays.append(arr.reshape(-1, 1))

	    return (
	        np.hstack(numeric_arrays) if numeric_arrays else np.array([]),
	        np.hstack(non_numeric_arrays).astype("O") if non_numeric_arrays else np.array([])
	    )
        
    def fit(self, X, y=None):
        
        num_feature, cat_feature = self.classify_arrays(X)
        if len(cat_feature) != 0:
            # self.one_hot_encoder.fit(cat_feature)
            self.imputer_cat.fit(cat_feature)
        if len(num_feature) != 0:
        	# self.standard_scaler.fit(num_feature)
        	self.imputer_num.fit(num_feature)

        return self
    
    def transform(self, X):
    	num_feature, cat_feature = self.classify_arrays(X)
    	# print(f"num: {num_feature}")
    	# print(f"cat_feature: {cat_feature}")
    	if len(num_feature) != 0:
    		num_feature = self.imputer_num.transform(num_feature)
    		# num_feature = self.standard_scaler.transform(num_feature)
    		# print(f"num_X: {X}")
    	if len(cat_feature) != 0:
    		# print(f"cat_bef: {cat_feature}")
    		cat_feature = self.imputer_cat.transform(cat_feature)
    		# print(f"cat_X: {X}")
    	X = [num_feature, cat_feature]
    		# print(f"final_X: {X}")
    	return X

class Encoder(BaseEstimator, TransformerMixin):
	
	def __init__(self):
		self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', 
											 drop="first")

	def fit(self, X, y=None):
		num_feature, cat_feature = X
		if len(cat_feature) != 0:
			self.one_hot_encoder.fit(cat_feature)
		return self

	def transform(self, X):
		num_feature, cat_feature = X
		if len(cat_feature) != 0:
			cat_feature = self.one_hot_encoder.transform(cat_feature)
			X = [num_feature, cat_feature]
		return X

class Scaler(BaseEstimator, TransformerMixin):

	def __init__(self):
		self.standard_scaler = StandardScaler()

	def fit(self, X, y=None):
		num_feature, cat_feature = X
		if len(num_feature) != 0:
			self.standard_scaler.fit(num_feature)

		return self

	def transform(self, X):
		num_feature, cat_feature = X

		if len(num_feature) != 0:
			num_feature = self.standard_scaler.transform(num_feature)
			X = num_feature
		if len(cat_feature)!=0 and len(num_feature)!=0:
			X = np.hstack([num_feature, cat_feature])
		elif len(cat_feature)!=0:
			X = cat_feature
		# print(f"final: {X}")
		return X



class Result:

	def __init__(self, feature, target, model_type, PCA=False, save=False):

		train_idx, test_idx = self.train_test_split_indices(len(target), test_size=0.2)

		
		X_train = [i[train_idx] for i in feature]
		y_train = target[train_idx]
		X_test = [i[test_idx] for i in feature]
		y_test = target[test_idx]

		pipe = self.transform_features(PCA)
		pipe.fit(X_train)

		X_train = pipe.transform(X_train)
		X_test = pipe.transform(X_test)

		self.all_data = {"X_test": X_test, "y_train":y_train, 
						 "X_train":X_train, "y_test":y_test}


	def train_test_split_indices(self, total_samples, test_size=0.2, random_seed=None):
		
		if random_seed is not None:
			np.random.seed(random_seed)

		test_indices = np.random.choice(total_samples, size=int(total_samples * test_size), replace=False)
		train_indices = np.setdiff1d(np.arange(total_samples), test_indices)

		return train_indices, test_indices

	def transform_features(self, PCA):
		pipe = [("Parse and impute", StructuredArrImputer()),
				("One hot encoding", Encoder()),
				("Feature scaling", Scaler())]
		if PCA:
			pipe.append(("PCA", PCA()))

		return Pipeline(pipe)

	def get_models(self):
		models = {"Linear Regression": LinearRegression(), 
			      "Ridge Regression": Ridge()}

		fittables = {i:FinalAlgo(j, self.all_data) for (i,j) in models.items()}
		return fittables

