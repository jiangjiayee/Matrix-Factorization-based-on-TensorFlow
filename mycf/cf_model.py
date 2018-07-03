import numpy as np
import tensorflow as tf

class CFModel(object):

	def __init__(self, shape, learning_rate, latent_factor, regularization_rate, loss, implicit, random_state = 0):
		
		self.shape = shape
		self.learning_rate = learning_rate
		self.latent_factor = latent_factor
		self.loss = loss
		self.implicit = implicit
		self.random_state = random_state
		self.regularization_rate = regularization_rate

		# the R (n, i) matrix is factorized to P (n, k) and Q (i, k) matrices
		n, i, k = self.shape

		# initialize the graph
		self.graph = tf.Graph()

		with self.graph.as_default():

			tf.set_random_seed(self.random_state)

			with tf.name_scope('inputs'):

				self.user_ids = tf.placeholder(tf.int32, shape=[None], name='user_ids')
				self.item_ids = tf.placeholder(tf.int32, shape=[None], name='item_ids')
				self.ratings = tf.placeholder(tf.float32, shape=[None], name='ratings')

				# if self.implicit:
				# 	targets = tf.clip_by_value(self.ratings, 0, 1, name='targets')
				# else:
				targets = tf.identity(self.ratings)


			with tf.name_scope('parameters'):

				self.user_embeddings = tf.get_variable('user_embeddings', shape=[n, k], dtype=tf.float32,
													initializer = tf.random_normal_initializer(mean=0, stddev=0.01))

				self.item_embeddings = tf.get_variable('item_embeddings', shape=[i, k], dtype=tf.float32,
													initializer = tf.random_normal_initializer(mean=0, stddev=0.01))

				self.user_bias = tf.get_variable('user_bias', shape=[n], dtype=tf.float32,
													initializer=tf.zeros_initializer())
				self.item_bias = tf.get_variable('item_bias', shape=[i], dtype=tf.float32,
													initializer=tf.zeros_initializer())

				self.global_bias = tf.get_variable('global_bias', shape=[], dtype=tf.float32,
													initializer=tf.zeros_initializer())

			with tf.name_scope('prediction'):
				# batch
				batch_user_bias = tf.nn.embedding_lookup(self.user_bias, self.user_ids)
				batch_item_bias = tf.nn.embedding_lookup(self.item_bias, self.item_ids)

				batch_user_embeddings = tf.nn.embedding_lookup(self.user_embeddings, self.user_ids)
				batch_item_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.item_ids)

				# P[u,:] * Q[i,:]
				temp_sum = tf.reduce_sum(tf.multiply(batch_user_embeddings, batch_item_embeddings), axis=1)

				bias = tf.add(batch_user_bias, batch_item_bias)
				bias = tf.add(bias, self.global_bias)

				predictor = tf.add(bias, temp_sum)
				if self.loss == 'logistic':
					self.pred = tf.sigmoid(predictor, name='predictions')
				else:
					self.pred = tf.identity(predictor, name='predictions')

				# all
				# temp_sum = tf.reduce_sum(tf.multiply(self.user_embeddings, self.item_embeddings), axis=1)
				# bias = tf.add(self.user_bias, self.item_bias)
				# bias = tf.add(bias, self.global_bias)

				# predictor = tf.add(bias, temp_sum)

				# self.pred = tf.identity(predictor, name='predictions')

			with tf.name_scope('loss'):

				l2_weights = tf.add(tf.nn.l2_loss(self.user_embeddings),tf.nn.l2_loss(self.item_embeddings))
				# 对偏置项加正则之后 效果变得很差
				# l2_bias = tf.add(tf.nn.l2_loss(self.user_bias),tf.nn.l2_loss(self.item_bias))
				# l2_term = tf.add(l2_weights, l2_bias)

				l2_term = l2_weights

				l2_term = tf.multiply(self.regularization_rate, l2_term, name='regularization')

				if self.loss == 'logistic':
					loss_raw = tf.losses.log_loss(predictions=self.pred, labels=targets)
				else:
					loss_raw = tf.losses.mean_squared_error(predictions=self.pred, labels=targets)

				self.cost = tf.add(loss_raw, l2_term)

				self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

			init = tf.global_variables_initializer()

		self.sess = tf.Session(graph=self.graph)
		self.sess.run(init)

	def train(self, users, items, ratings):
		batch = {
			self.user_ids:users,
			self.item_ids:items,
			self.ratings:ratings
		}
		_, loss_value = self.sess.run(fetches=[self.train_step, self.cost],
												feed_dict=batch)
		return loss_value


	def predict(self, users, items):
		batch = {
			self.user_ids : users,
			self.item_ids : items
		}
		return self.pred.eval(feed_dict=batch, session=self.sess)
