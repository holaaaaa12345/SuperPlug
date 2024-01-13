"""
API to connect to either custom backend or to sklearn. This API
consists of:
   1) Custom implementation of model classes by inheriting from the backend.
   2) FinalModel class to wrap the model classes s.t. they are ready
      to be fitted.
   3) PreprocessData class to clean and convert data.
"""
import numpy as np

"""
This API allows for seamless backend switching between custom (from scratch) 
and Scikit-Learn. To use Scikit-Learn, set the following variable to True.
You should use Scikit-Learn for large dataset (>2000 rows) to avoid freezing.

"""
USE_SKLEARN = False
if USE_SKLEARN:
	from sklearn_backend import *

else:
	from custom_backend import *


#######################################################
# The following are parent classes to evaluate models #
#######################################################

class CustomClassification:
	pass

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

#####################################################################
# The following are model classes that inherit from backend models #
#####################################################################

class LinearRegression(LinearRegression, CustomRegression):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._all_data = None

class Ridge(Ridge, CustomRegression):

	def __init__(self, alpha=1, **kwargs):
		super().__init__(alpha=alpha, **kwargs)
		self._all_data = None
		self._param_space = {"alpha": np.linspace(0.0001, 1, 100)}

class KNeighborsRegressor(KNeighborsRegressor, CustomRegression):

	def __init__(self, n_neighbors=5, **kwargs):
		super().__init__(n_neighbors=n_neighbors, **kwargs)
		self._all_data = None
		self._param_space = {"n_neighbors": np.arange(2, 13)}

##########################################################
# The following are classes to preprocess and clean data #
##########################################################

class StructuredArrImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
    	self.imputer_num = SimpleImputer(strategy="mean")
    	self.imputer_cat = SimpleImputer(strategy="most_frequent", missing_values="")
        
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


#####################################################################
# The following class and function will be called from the frontend #
#####################################################################

class FinalModel():
	"""
	Preprocessing class to wrap models to incorporate hyperparametric tuning
	(randomizedsearchCV). It can then be fitted like a normal model.
	"""
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

		# TODO: Surely there's a better way to do this -__-
		setattr(estimator_final, "_all_data", self._all_data)
		return estimator_final

class PreprocessData:
	""" 
	Class to preprocess the data, which includes the train-test-split,
	conversion from structured array to normal numpy array, imputation, encoding, 
	and scaling.
	"""

	def __init__(self, model_type=None, run_pca=False):
		self.model_type = model_type
		self.run_pca = run_pca

	def train_test_split_indices(self, total_samples, test_size=0.2, random_seed=None):
		
		if random_seed is not None:
			np.random.seed(random_seed)

		test_indices = np.random.choice(total_samples, size=int(total_samples * test_size), replace=False)
		train_indices = np.setdiff1d(np.arange(total_samples), test_indices)

		return train_indices, test_indices

	def get_pipeline(self):

		pipe = [("Parse and impute", StructuredArrImputer()),
				("One hot encoding", Encoder()),
				("Feature scaling", Scaler())]
		if self.run_pca:
			pipe.append(("PCA", PCA()))

		return Pipeline(pipe)

	def get_final_data(self, feature, target):
		"""
		Method to return the all data, which includes train and test
		data, in form of a dictionary.
		"""
		train_idx, test_idx = self.train_test_split_indices(len(target), test_size=0.2)
		
		X_train = [i[train_idx] for i in feature]
		y_train = target[train_idx]
		X_test = [i[test_idx] for i in feature]
		y_test = target[test_idx]

		pipe = self.get_pipeline()
		pipe.fit(X_train)

		X_train = pipe.transform(X_train)
		X_test = pipe.transform(X_test)

		all_data = {"X_test": X_test, "y_train":y_train, 
				    "X_train":X_train, "y_test":y_test}

		return all_data


# Stand alone function to get all the models available given the type and all data.
def get_models(model_type, all_data):

	if model_type=="Regression":
		models = {"Linear Regression": LinearRegression(), 
			      "Ridge Regression": Ridge(),
			      "KNN Regression": KNeighborsRegressor()}
	else:
		models = {}

	# Wrap all the models in PreprocessModel class
	fittables = {i:FinalModel(j, all_data) for (i,j) in models.items()}
	return fittables

