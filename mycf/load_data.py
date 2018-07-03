import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import random
from collections import defaultdict

def load_data(path):
	# with open(path,'r') as f:
	# 	data = f.readlines()
	# 	print(len(data))
	df = pd.read_csv(path)

	rows, user_2_id = norm(df['userId'])
	cols, movie_2_id = norm(df['movieId'])
	ratings = df['rating'].values

	n = np.max(rows) + 1
	k = np.max(cols) + 1

	coo = coo_matrix((ratings, (rows, cols)), shape=(n,k))

	return coo.todok()

def norm(series):
	# 给ID 重新命名，因为movie中很多movieId没有出现过
	# 所以这样来减少矩阵维度
	index_dic = dict()
	count = 0
	for i in series.unique():
		index_dic[i] = count
		count += 1
	return series.apply(lambda x:index_dic[x]).values, index_dic

def train_test_split(path, frac, random_state=1, implicit=False):

	df = pd.read_csv(path)

	rows, user_2_id = norm(df['userId'])
	cols, movie_2_id = norm(df['movieId'])
	if implicit:
		df['rating'] = 1
		ratings = df['rating']
	else:
		ratings = df['rating'].values

	n = np.max(rows) + 1
	k = np.max(cols) + 1

	train_idx = df.sample(frac=frac,random_state=random_state).index.values
	test_idx = test_idx = np.array(list(set(range(len(ratings))) - set(train_idx)))

	train_data = coo_matrix((ratings[train_idx], (rows[train_idx], cols[train_idx])), shape=(n,k))
	test_data = coo_matrix((ratings[test_idx], (rows[test_idx], cols[test_idx])), shape=(n,k))

	return train_data.todok(), test_data.todok()

def train_test_split_changed(path, frac, random_state=1, implicit=False):

	# ml-latest-small
	if path == 'data/ml-latest-small/ratings.csv':
		df = pd.read_csv(path)
	else:
	# ml-1m
		df = pd.read_csv(path,sep='::',names=['userId','movieId','rating','timestamp'],engine='python')

	rows, user_2_id = norm(df['userId'])
	cols, movie_2_id = norm(df['movieId'])
	if implicit:
		df['rating'] = 1
		ratings = df['rating']
	else:
		ratings = df['rating'].values

	n = np.max(rows) + 1
	k = np.max(cols) + 1

	# 原始样本中 训练集 测试集的正样本ID
	train_idx = df.sample(frac=frac,random_state=random_state).index.values
	test_idx = test_idx = np.array(list(set(range(len(ratings))) - set(train_idx)))

	print('原始样本划分完毕')
	if implicit:
		# 隐式反馈 需要自行构建负样本

		# 为构建电影池做辅助
		movieId_value_counts = df['movieId'].value_counts()
		movieId_value_counts_index = df['movieId'].value_counts().index

		# 构建带权重的电影池 更热门的电影会有更多数量
		movieIdPool = []
		for movieId in movieId_value_counts_index:
			for i in range(movieId_value_counts[movieId]):
				movieIdPool.append(movieId)

		

		# 所有的 用户-电影 交互信息
		user_item_dic = get_interaction(df)

		# --------------开始构建训练集----------------------------
		
		# train_positive = get_interaction(df.loc[train_idx])
		# train_negative = defaultdict(list)

		# # 构建 训练集 的负样本，保证两点
		# # 1、同一个用户正负样本数量相同
		# # 2、用户对热门电影没有评价，更能说明不喜欢，所以可作为负样本
		# for user in df.loc[train_idx]['userId'].unique():
		# 	for i in range(len(train_positive[user])):
		# 		item = random.choice(movieIdPool)
		# 		# 负样本必然不在 原始评分表 中，
		# 		# 且 负样本不重复
		# 		while item in user_item_dic[user] or item in train_negative[user]:
		# 			item = random.choice(movieIdPool)
		# 		train_negative[user].append(item)

		# users, movies, ratings = get_input_array(train_positive, train_negative, user_2_id, movie_2_id,train=True)
		# train_data = coo_matrix((ratings,(users,movies)),shape=(n,k))
		train_data = coo_matrix((ratings[train_idx], (rows[train_idx], cols[train_idx])), shape=(n,k))
		print('train_data completed')
		# --------------开始构建测试集----------------------------
		# 测试集 中的 正例
		test_positive = get_interaction(df.loc[test_idx])
		test_negative = defaultdict(list)
		# 构建 测试集 的负样本，保证两点
		# 1、同一个用户正负样本数量相同
		# 2、用户对热门电影没有评价，更能说明不喜欢，所以可作为负样本
		for user in df.loc[test_idx]['userId'].unique():
			for i in range(len(test_positive[user])):
				item = random.choice(movieIdPool)
				# 负样本必然不在 原始评分表中（因为原始评分表一部分是训练集，另一部分是测试集的正例）
				# 且 不在训练集的负样本中
				# 且 负样本不重复
				while item in user_item_dic[user] or item in test_negative[user]:
					item = random.choice(movieIdPool)
				test_negative[user].append(item)

		users, movies, ratings = get_input_array(test_positive, test_negative, user_2_id, movie_2_id,train=False)
		test_data = coo_matrix((ratings,(users,movies)),shape=(n,k))
		print('test_data completed')

		return train_data.todok(), test_data.todok()
	else:
		# 显式反馈，不需要额外构建负样本
		train_data = coo_matrix((ratings[train_idx], (rows[train_idx], cols[train_idx])), shape=(n,k))
		test_data = coo_matrix((ratings[test_idx], (rows[test_idx], cols[test_idx])), shape=(n,k))

		return train_data.todok(), test_data.todok()

def train_test_split_third(path, frac=0.8, random_state=1, implicit=False):
	# ml-latest-small
	if path == 'data/ml-latest-small/ratings.csv':
		df = pd.read_csv(path)
	else:
	# ml-1m
		df = pd.read_csv(path,sep='::',names=['userId','movieId','rating','timestamp'],engine='python')

	users, user_2_id = norm(df['userId'])
	movies, movie_2_id = norm(df['movieId'])
	if implicit:
		df['rating'] = 1
		ratings = df['rating']
	else:
		ratings = df['rating'].values

	n = np.max(users) + 1
	k = np.max(movies) + 1

	if not implicit: 
		# 显式反馈，不需要额外构建负样本

		# 原始样本中 训练集 测试集的正样本ID
		train_idx = df.sample(frac=frac,random_state=random_state).index.values
		test_idx = test_idx = np.array(list(set(range(len(ratings))) - set(train_idx)))

		train_data = coo_matrix((ratings[train_idx], (users[train_idx], movies[train_idx])), shape=(n,k))
		test_data = coo_matrix((ratings[test_idx], (users[test_idx], movies[test_idx])), shape=(n,k))

		return train_data.todok(), test_data.todok(), ''

	else:
		# 隐式反馈，需要构建额外负样本
		initial_matrix = coo_matrix((ratings, (users, movies)), shape=(n,k)).tocsr()

		test_matrix = initial_matrix.copy()
		train_matrix = initial_matrix.copy()
		nonzero_pairs = list(zip(users, movies)) # Zip these pairs together of user,item index into list
		random.seed(0) # Set the random seed to zero for reproducibility
		num_samples = int(np.ceil((1-frac)*len(nonzero_pairs))) # Round the number of samples needed to the nearest integer
		samples = random.sample(nonzero_pairs, num_samples) # Sample a random number of user-item pairs without replacement

		user_inds = [index[0] for index in samples] # Get the user row indices
		item_inds = [index[1] for index in samples] # Get the item column indices

		train_matrix[user_inds, item_inds] = 0 # Assign all of the randomly chosen user-item pairs to zero
		train_matrix.eliminate_zeros() # Get rid of zeros in sparse array storage after update to save space

		return train_matrix.todok(), test_matrix.todok(), user_inds


def record(x,dic):
    dic[x.userId].append(x.movieId)
    return 1

def get_interaction(df):
	dic = defaultdict(list)
	a = df.apply(lambda x:record(x,dic),axis=1)
	return dic

def get_input_array(positive, negative, user_2_id, movie_2_id,train=False):
	users = []
	movies = []
	ratings = []

	for user in positive:
		for index in range(len(positive[user])):
			users.append(user_2_id[user])
			users.append(user_2_id[user])
			movies.append(movie_2_id[positive[user][index]])
			movies.append(movie_2_id[negative[user][index]])
			if train:
				ratings.append(1)
				ratings.append(0)
			else:
				ratings.append(2)
				ratings.append(1)
	return np.array(users), np.array(movies), np.array(ratings)

if __name__ == '__main__':
	load_data('data/ml-latest-small/ratings.csv')