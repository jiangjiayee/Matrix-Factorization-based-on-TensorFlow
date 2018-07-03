import numpy as np

# from mycf.gcn_mf import GCNMF
# from mycf.cf_model import CFModel
from mycf import GCNMF, sparse_matrix
from mycf.load_data import load_data, train_test_split_changed, train_test_split_third

# user_id = [0,0,1,1,2,2]
# movie_id = [0,1,2,0,1,2]
# rating = [1,1,2,2,3,3]
# X = sparse_matrix(user_id, movie_id, rating)
implicit = True

# X = load_data('data/ml-latest-small/ratings.csv')
# data/ml-1m/ratings.dat
train_data, test_data, user_idx = train_test_split_third('data/ml-1m/ratings.dat',frac=0.8,random_state=1,implicit=implicit)
print(train_data.toarray())
print(test_data.toarray())
mf = GCNMF(latent_factor=5, batch_size=500,n_iter=2000,implicit=implicit)
mf.fit(train_data)
mf.cal_test(test_data,user_idx,implicit=implicit)
# print(loss)
# mf.fit(train_data, test_data)
# train_loss, test_loss = mf.get_losses()
# print(train_loss)
# print(test_loss)
# print(mf.predict_all().A)
# mf.print_history()