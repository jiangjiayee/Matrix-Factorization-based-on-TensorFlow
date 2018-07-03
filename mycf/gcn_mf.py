import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import metrics

from tqdm import trange

from .cf_model import CFModel
class GCNMF(object):

	def __init__(self, latent_factor=5, batch_size=500, n_iter=500,
				learning_rate=0.01, regularization_rate=0.02,implicit=False):

		self.latent_factor = latent_factor
		self.shape = (None, None, self.latent_factor)
		
		self._tf = None
		self.batch_size = batch_size
		self.n_iter = n_iter
		self.learning_rate = float(learning_rate)
		self.regularization_rate = regularization_rate
		self.implicit = implicit
		# self.warm_start = warm_start
		if self.implicit == False:
			self.loss = 'MSE'
		else:
			self.loss = 'logistic'
		self._fresh_session()


	def _fresh_session(self):
	# reset the session, to start from the scratch        
		self._tf = None
		self.train_loss = []
		self.test_loss = []

	def init_with_shape(self, u, i):
		self.shape = (int(u), int(i), int(self.latent_factor))
		self._tf_init()

	def _tf_init(self):
		self._tf = CFModel(shape=self.shape, learning_rate=self.learning_rate, latent_factor=self.latent_factor, 
							regularization_rate=self.regularization_rate, loss=self.loss, implicit=self.implicit)

	def fit(self, sparse_matrix, test=None):
		# 
		# Fit the model

		# Fit the model starting at randomly initialized parameters. When
		# warm_start=True, this method works the same as partial_fit.

		# Parameters
		# ----------

		# sparse_matrix : sparse-matrix, shape (n_users, n_items)
		# 	Sparse matrix in scipy.sparse format, can be created using sparse_matrix
		# 	function from this package.
		# 
		# if not self.warm_start:
		# 	self._fresh_session()
		if test != None:
			return self.testloss_fit(sparse_matrix, test)
		else:
			return self.partial_fit(sparse_matrix)

	def partial_fit(self, sparse_matrix):
		"""Fit the model

		Fit the model starting at previously trained parameter values. If the
		model was not trained yet, it randomly initializes parameters same as
		the fit method.

		Parameters
		----------

		sparse_matrix : sparse-matrix, shape (n_users, n_items)
			Sparse matrix in scipy.sparse format, can be created using sparse_matrix
			function from this package.
		"""
		if self._tf is None:
			self.init_with_shape(*sparse_matrix.shape)

		batch = self._batch_generator(sparse_matrix, size=self.batch_size, nonzero=not self.implicit)

		for _ in trange(self.n_iter):
			batch_users, batch_items, batch_ratings = next(batch)
			loss_value = self._tf.train(batch_users, batch_items, batch_ratings)
			self.train_loss.append(loss_value)
		
		return self

	def testloss_fit(self, sparse_matrix, test):
		if self._tf is None:
			self.init_with_shape(*sparse_matrix.shape)

		batch = self._batch_generator(sparse_matrix, size=self.batch_size, nonzero=not self.implicit)

		for _ in trange(self.n_iter):
			batch_users, batch_items, batch_ratings = next(batch)
			train_loss_value = self._tf.train(batch_users, batch_items, batch_ratings)
			self.train_loss.append(train_loss_value)

			if _ % 10 == 0:
				test_loss_value = self.cal_test(test)
				self.test_loss.append(test_loss_value)
		
		return self

	def _batch_generator(self, data, size=1, nonzero=False):
		if nonzero:
			# for explicit ratings
			users, items = data.nonzero()
			while True:
				idx = np.random.randint(len(users), size=size)
				yield users[idx], items[idx], data[users[idx], items[idx]].A.flatten()
		else:
			# for implicit ratings
			while True:
				users = np.random.randint(self.shape[0], size=size)
				items = np.random.randint(self.shape[1], size=size)
				vals = data[users, items].A.flatten()
				yield users, items, vals



	def get_losses(self):
		return self.train_loss, self.test_loss

	def predict(self, users, items):
		"""Predict using the model

		Parameters
		----------

		rows : array, shape (n_samples,)
			Make predictions for those row indexes. If not provided,
			makes predictions for all the possible rows (use with caution).

		cols : array, shape (n_samples,)
			Make predictions for those  column indexes. If not provided,
			makes predictions for all the possible columns (use with caution).

		Returns
		-------
		array, shape (n_samples,)
			Predictions for given indexes.
		"""
		return self._tf.predict(users,items)

	def cal_test(self, test_matrix, user_inds, implicit=True):
		

		

		if implicit:

			user_inds = list(set(user_inds))
			y_ = test_matrix[user_inds,:].A.flatten()
			print('y_ is:{}'.format(y_))
			print('Sum of y_ is:{}'.format(sum(y_)))
			print('Length of y_ is:{}'.format(len(y_)))
			items = np.array([x for x in range(self.shape[1])] * len(user_inds))
			users = np.repeat(user_inds, self.shape[1])
			pred = self.predict(users,items)

			# y_ = test_matrix[users, items].A.flatten() - 1

			print('The pred result is:{}'.format(pred))
			test_auc = metrics.roc_auc_score(y_, pred)
			print('The AUC score of test_data is:{}'.format(test_auc))

		else:
			users,items = test_matrix.nonzero()

			pred = self.predict(users, items)
			y_ = test_matrix[users, items].A.flatten()

			MSE = mean_squared_error(pred, y_)
			MAE = mean_absolute_error(pred, y_)
			print('Emplicit calculate, the MSE is {}'.format(MSE))
			print('Emplicit calculate, the MAE is {}'.format(MAE))
		# print(pred)
		# print(y_)
		# print(MSE)

		# return MSE