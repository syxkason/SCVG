"""
Paper: Semi-deterministic and Contrastive Variational Graph Autoencoder for Recommendation
Author: Yue Ding, Yuxiang Shi, Bo Chen, Chenghua Lin, Hongtao Lu, Jie Li, Ruiming Tang, Dong Wang
Reference: https://github.com/syxkason/SCVG
"""

import scipy.sparse as sp
import tensorflow as tf
import numpy as np
from model.AbstractRecommender import AbstractRecommender
import heapq
import sys
from concurrent.futures import ThreadPoolExecutor
import itertools
from time import time

class SCVG(AbstractRecommender):
	def __init__(self, sess, dataset, config):
		super(SCVG, self).__init__(dataset, config)
		self.lr = config['lr']
		self.embedding_size = config['embedding_size']
		self.epochs = config["epochs"]
		self.keep_prob = config['keep_prob']
		self.dropout = config['dropout']
		self.n_hidden = config['n_hidden']
		self.dataset = dataset
		self.Ks = [5, 10, 20, 30]
		self.n_fold = 8192
		self.n_users, self.n_items = self.dataset.num_users, self.dataset.num_items
		self.user_pos_train = self.dataset.get_user_train_dict(by_time=False)
		self.all_users = list(self.user_pos_train.keys())
		self.path = 'dataset/'
		self.sess = sess
		self.n_nodes = self.dataset.num_users + self.dataset.num_items
		self.shape = np.array([self.n_nodes, self.n_nodes])
		self.norm_adj_mat_real, self.pre_adj_mat_real = self.get_adj_mat()
		self.norm_adj_mat = tf.sparse_placeholder(tf.float32, shape=[None, self.n_items], name='norm_adj_mat')
		self.pre_adj_mat = tf.sparse_placeholder(tf.float32, shape=[None, self.n_items], name='pre_adj_mat')


		self.keep_prob_variable = tf.placeholder(tf.float32)
		self.is_training = tf.placeholder(tf.bool, name='is_training')
		self.mu_np = []
		self.users = tf.placeholder(tf.int32)

		self.W_0_mu = unif_weight_init(shape=[self.n_nodes, self.n_hidden])
		self.b_0_mu = tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[self.n_hidden]))
		self.W_1_mu = unif_weight_init(shape=[self.n_hidden, self.embedding_size])
		self.b_1_mu = tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[self.embedding_size]))

		self.W_0_mlp = unif_weight_init(shape=[self.n_hidden + self.embedding_size, 20])
		self.b_0_mlp = tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[20]))
		self.W_1_mlp = unif_weight_init(shape=[20, 1])
		self.b_1_mlp = tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[1]))

		self.weights = dict()
		initializer = tf.random_normal_initializer(stddev=0.01)
		self.weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.n_hidden]), name='user_embedding')
		self.weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.n_hidden]), name='item_embedding')

		self.dim_weight = 0.2
		self._build_VGAE()
		self.lz_loss = tf.reduce_mean(tf.square(self.mu)) / self.n_nodes * 1 / 5
		self.reconstruction_loss = self.ComputeVaeLoss()
		self.dim_loss = self.DIM()
		self.vgae_loss = self.lz_loss + self.reconstruction_loss + self.dim_loss
		self.optimizer = tf.train.AdamOptimizer(self.lr)
		self.train_step = self.optimizer.minimize(self.vgae_loss)

		init = tf.global_variables_initializer()
		self.sess.run(init)

	def ComputeVaeLoss(self):
		A_fold_hat = self._convert_sp_mat_to_sp_tensor(self.pre_adj_mat_real)
		self.z = self.sample_gaussian(A_fold_hat, self.mu)
		self.u_z_embeddings = self.z[:self.n_users, :]
		self.i_z_embeddings = self.z[self.n_users:, :]
		self.u_z_embeddings_fold = tf.nn.embedding_lookup(self.u_z_embeddings, self.users)
		self.batch_rating_vae = tf.matmul(self.u_z_embeddings_fold, self.i_z_embeddings, transpose_a=False, transpose_b=True)
		z_u_pred_logtis = tf.clip_by_value(self.batch_rating_vae, 1e-10, 10.0)
		z_u_pred_logtis = tf.reshape(z_u_pred_logtis, [-1, self.n_items])
		y_label = self.norm_adj_mat
		dense_y_label = tf.sparse_tensor_to_dense(sp_input=y_label, default_value=0, validate_indices=False, name=None)
		dense_y_label_nitem = dense_y_label[:, self.n_users:]
		z_u_pred_logtis = tf.reshape(z_u_pred_logtis, [-1, self.n_items])
		w_1 = (self.n_nodes * self.n_nodes - tf.reduce_sum(dense_y_label_nitem)) / tf.reduce_sum(dense_y_label_nitem)
		w_2 = self.n_nodes * self.n_nodes / ((self.n_nodes * self.n_nodes - tf.reduce_sum(dense_y_label_nitem)) * 2)
		cross_entropy = w_2 * tf.reduce_mean(
			(1 - dense_y_label_nitem) * z_u_pred_logtis + (1 + (w_1 - 1) * dense_y_label_nitem) * tf.log(
				1 + tf.exp(-z_u_pred_logtis)))
		return cross_entropy

	def get_adj_mat(self):
		try:
			adj_mat = sp.load_npz(self.path + 'adj_mat_yelp.npz')
			pre_adj_mat = sp.load_npz(self.path + 'pre_adj_mat_yelp.npz')
			print('already load adj matrix', adj_mat.shape)
		except Exception:
			adj_mat, pre_adj_mat = self.create_adj_mat()
			sp.save_npz(self.path + 'adj_mat_yelp.npz', adj_mat)
			sp.save_npz(self.path + 'pre_adj_mat_yelp.npz', pre_adj_mat)
		return adj_mat, pre_adj_mat

	def create_adj_mat(self):

		user_list, item_list = self.dataset.get_train_interactions()
		user_np = np.array(user_list, dtype=np.int32)
		item_np = np.array(item_list, dtype=np.int32)
		ratings = np.ones_like(user_np, dtype=np.float32)
		R1 = sp.csr_matrix((ratings, (user_np, item_np)), shape=(self.n_users, self.n_items))

		self.R = R1.todok()
		adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
		adj_mat = adj_mat.tolil()
		R = self.R.tolil()
		for i in range(5):
			adj_mat[int(self.n_users * i / 5.0):int(self.n_users * (i + 1.0) / 5), self.n_users:] = \
				R[int(self.n_users * i / 5.0):int(self.n_users * (i + 1.0) / 5)]
			adj_mat[self.n_users:, int(self.n_users * i / 5.0):int(self.n_users * (i + 1.0) / 5)] = \
				R[int(self.n_users * i / 5.0):int(self.n_users * (i + 1.0) / 5)].T
		adj_mat = adj_mat.todok()

		def normalized_adj_single(adj):
			rowsum = np.array(adj.sum(1))
			d_inv = np.power(rowsum, -0.5).flatten()
			d_inv[np.isinf(d_inv)] = 0.
			d_mat_inv = sp.diags(d_inv)
			norm_adj = d_mat_inv.dot(adj)
			print('generate normalized adjacency matrix.')
			return norm_adj.tocoo()

		norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
		pre_adj_mat = normalized_adj_single(adj_mat)
		return norm_adj_mat.tocsr(), pre_adj_mat.tocsr()

	def _build_VGAE(self):
		A_fold_hat = self._convert_sp_mat_to_sp_tensor(self.norm_adj_mat_real)
		hidden_0_mu_ = gcn_layer_id(A_fold_hat, self.W_0_mu, self.b_0_mu)
		if self.dropout:
			hidden_0_mu = tf.nn.dropout(hidden_0_mu_, self.keep_prob_variable)
		else:
			hidden_0_mu = hidden_0_mu_
		self.mu = gcn_layer(A_fold_hat, hidden_0_mu, self.W_1_mu, self.b_1_mu)
		return self.mu

	def DIM(self):
		neg_z = tf.gather(self.z, tf.random_shuffle(tf.range(tf.shape(self.z)[0])))

		pos_embed = tf.concat([self.W_0_mu, self.z], 1)
		neg_embed = tf.concat([self.W_0_mu, neg_z], 1)
		pos_embeddings = tf.nn.relu(tf.matmul(pos_embed, self.W_0_mlp) + self.b_0_mlp)
		pos_embeddings = tf.nn.sigmoid(tf.matmul(pos_embeddings, self.W_1_mlp) + self.b_1_mlp)
		neg_embeddings = tf.nn.relu(tf.matmul(neg_embed, self.W_0_mlp) + self.b_0_mlp)
		neg_embeddings = tf.nn.sigmoid(tf.matmul(neg_embeddings, self.W_1_mlp) + self.b_1_mlp)
		pos_embeddings = tf.clip_by_value(pos_embeddings, 1e-10, 10.0)
		neg_embeddings = tf.clip_by_value(neg_embeddings, 1e-10, 10.0)

		DIM_loss = - tf.reduce_mean((tf.log_sigmoid(pos_embeddings)) + tf.log(1 - tf.nn.sigmoid(neg_embeddings))) / 2
		return DIM_loss * self.dim_weight

	def _convert_sp_mat_to_sp_tensor(self, X):
		coo = X.tocoo().astype(np.float32)
		indices = np.mat([coo.row, coo.col]).transpose()
		return tf.SparseTensor(indices, coo.data, coo.shape)

	def _convert_sp_mat_to_list(self, X):
		coo_adjacency = X.tocoo().astype(np.float32)
		indices = np.vstack((coo_adjacency.row, coo_adjacency.col)).transpose()
		values = coo_adjacency.data
		shape = np.array(coo_adjacency.shape, dtype=np.int64)

		return (indices, values, shape)

	def sample_gaussian(self, adj_mat, mean):
		z = mean + tf.random_normal(mean.shape, mean=0.0,
									stddev=tf.sparse_tensor_dense_matmul(adj_mat, mean) / self.n_nodes)
		return z

	def train_model(self):

		users = list(i for i in range(self.n_users))
		feed_dict = {
			self.norm_adj_mat: self._convert_sp_mat_to_list(self.norm_adj_mat_real[:self.n_users]),
			self.pre_adj_mat: self._convert_sp_mat_to_list(self.pre_adj_mat_real[:self.n_users]),
			self.keep_prob_variable: self.keep_prob,
			self.users: users}

		cur_best_pre_0 = 0
		stopping_step = 0
		should_stop = False

		pre_loger, rec_loger, ndcg_loger, mrr_loger = [], [], [], []

		for i in range(1, self.epochs + 1):
			_, loss, reconst_loss, lz_loss, self.mu_np, dim_loss= \
				self.sess.run([self.train_step,
							   self.vgae_loss,
							   self.reconstruction_loss,
							   self.lz_loss,
							   self.mu,
							   self.dim_loss
							   ],
							  feed_dict=feed_dict)
			self.logger.info("[iter %d : rescon_loss : %f , lz_loss : % f , dim_loss : %f , loss : %f]" % (i, reconst_loss, lz_loss, dim_loss, loss))

			print("At step {0}  Loss: {1}   Reconst: {2} Lz: {3} dim_loss: {4} ".format(i, loss, reconst_loss, lz_loss, dim_loss))

			if (i % 1 == 0 and i >= 600):
				top_show = np.sort(self.Ks)
				max_top = max(top_show)
				result = {'precision': np.zeros(len(self.Ks)), 'recall': np.zeros(len(self.Ks)),
						  'ndcg': np.zeros(len(self.Ks)), 'mrr': np.zeros(len(self.Ks))}

				u_batch_size = self.n_fold
				test_dict = self.dataset.get_user_test_dict()
				subkey = test_dict.keys()
				test = dict([(key, test_dict[key]) for key in subkey])

				train_dict = self.dataset.get_user_train_dict()
				subkey1 = train_dict.keys()
				train = dict([(key, train_dict[key]) for key in subkey1])

				test_users = list(test.keys())
				n_test_users = len(test_users)
				n_user_batchs = n_test_users // u_batch_size + 1

				count = 0
				all_result = []

				for u_batch_id in range(n_user_batchs):
					start = u_batch_id * u_batch_size
					end = (u_batch_id + 1) * u_batch_size

					user_batch = test_users[start: end]

					rate_batch = self.sess.run(self.batch_rating_vae, feed_dict={self.users: user_batch,
																				 self.keep_prob_variable: 1
																				 })
					rate_batch = np.array(rate_batch)
					rate_batch = np.reshape(rate_batch, [-1, self.n_items])

					test_items = []
					for user in user_batch:
						test_items.append(test[user])  # (B, #test_items)

					for idx, user in enumerate(user_batch):
						train_items_off = train[user]
						rate_batch[idx][train_items_off] = -np.inf

					batch_result = eval_score_matrix_foldout(rate_batch, test_items,
															 max_top)  # (B,k*metric_num), max_top= 20
					count += len(batch_result)
					all_result.append(batch_result)
					print(count)

				assert count == n_test_users
				all_result = np.concatenate(all_result, axis=0)  # user * (5*max_top)
				final_result = np.mean(all_result, axis=0)  # mean
				final_result = np.reshape(final_result, newshape=[5, max_top])
				final_result = final_result[:, top_show - 1]
				final_result = np.reshape(final_result, newshape=[5, len(top_show)])
				result['precision'] += final_result[0]
				result['recall'] += final_result[1]
				result['ndcg'] += final_result[3]
				result['mrr'] += final_result[4]

				ret = 'model recall=[%s], precision=[%s], ' \
					  'ndcg=[%s], mrr=[%s]' % \
					  (', '.join(['%.5f' % r for r in result['recall']]),
					   ', '.join(['%.5f' % r for r in result['precision']]),
					   ', '.join(['%.5f' % r for r in result['ndcg']]),
					   ', '.join(['%.5f' % r for r in result['mrr']]))
				print(ret + '\n')

				self.logger.info("%s" % ret + '\n')

				rec_loger.append(result['recall'])
				pre_loger.append(result['precision'])
				ndcg_loger.append(result['ndcg'])
				mrr_loger.append(result['mrr'])

				cur_best_pre_0, stopping_step, should_stop = early_stopping(result['recall'][2], cur_best_pre_0,
																			stopping_step, flag_step=20)

			if should_stop == True:
				break

		recs = np.array(rec_loger)
		pres = np.array(pre_loger)
		ndcgs = np.array(ndcg_loger)
		mrr = np.array(mrr_loger)

		best_rec_0 = max(recs[:, 2])
		idx = list(recs[:, 2]).index(best_rec_0)

		final_perf = "Best Iter=[%d] \trecall=[%s], precision=[%s], ndcg=[%s], mrr=[%s]" % \
					 (idx, '\t'.join(['%.5f' % r for r in recs[idx]]),
					  '\t'.join(['%.5f' % r for r in pres[idx]]),
					  '\t'.join(['%.5f' % r for r in ndcgs[idx]]),
					  '\t'.join(['%.5f' % r for r in mrr[idx]]))
		print(final_perf)
		self.logger.info("%s" % final_perf + '\n')

def unif_weight_init(shape, name=None):
	initial = tf.random_uniform(shape, minval=-np.sqrt(6.0 / (shape[0] + shape[1])),
								maxval=np.sqrt(6.0 / (shape[0] + shape[1])), dtype=tf.float32)
	return tf.Variable(initial, name=name)


def gcn_layer_id(norm_adj_mat, W, b):
	return tf.nn.relu(tf.add(tf.sparse_tensor_dense_matmul(norm_adj_mat, W), b))


def gcn_layer(norm_adj_mat, h, W, b):
	return tf.add(tf.matmul(tf.sparse_tensor_dense_matmul(norm_adj_mat, h), W), b)


def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))


def argmax_top_k(a, top_k=50):
	ele_idx = heapq.nlargest(top_k, zip(a, itertools.count()))
	return np.array([idx for ele, idx in ele_idx], dtype=np.intc)


def precision(rank, ground_truth):
	hits = [1 if item in ground_truth else 0 for item in rank]
	result = np.cumsum(hits, dtype=np.float) / np.arange(1, len(rank) + 1)
	return result


def recall(rank, ground_truth):
	hits = [1 if item in ground_truth else 0 for item in rank]
	result = np.cumsum(hits, dtype=np.float) / len(ground_truth)
	return result


def map(rank, ground_truth):
	pre = precision(rank, ground_truth)
	pre = [pre[idx] if item in ground_truth else 0 for idx, item in enumerate(rank)]
	sum_pre = np.cumsum(pre, dtype=np.float32)
	gt_len = len(ground_truth)
	# len_rank = np.array([min(i, gt_len) for i in range(1, len(rank)+1)])
	result = sum_pre / gt_len
	return result


def ndcg(rank, ground_truth):
	len_rank = len(rank)
	len_gt = len(ground_truth)
	idcg_len = min(len_gt, len_rank)

	# calculate idcg
	idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
	idcg[idcg_len:] = idcg[idcg_len - 1]

	# idcg = np.cumsum(1.0/np.log2(np.arange(2, len_rank+2)))
	dcg = np.cumsum([1.0 / np.log2(idx + 2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
	result = dcg / idcg
	return result


def mrr(rank, ground_truth):
	last_idx = sys.maxsize
	for idx, item in enumerate(rank):
		if item in ground_truth:
			last_idx = idx
			break
	result = np.zeros(len(rank), dtype=np.float32)
	result[last_idx:] = 1.0 / (last_idx + 1)
	return result


#
def eval_score_matrix_foldout(score_matrix, test_items, top_k=50, thread_num=None):
	def _eval_one_user(idx):
		scores = score_matrix[idx]  # all scores of the test user
		test_item = test_items[idx]  # all test items of the test user

		ranking = argmax_top_k(scores, top_k)  # Top-K items
		result = []
		result.extend(precision(ranking, test_item))
		result.extend(recall(ranking, test_item))
		result.extend(map(ranking, test_item))
		result.extend(ndcg(ranking, test_item))
		result.extend(mrr(ranking, test_item))

		result = np.array(result, dtype=np.float32).flatten()
		return result

	with ThreadPoolExecutor(max_workers=thread_num) as executor:
		batch_result = executor.map(_eval_one_user, range(len(test_items)))

	result = list(batch_result)  # generator to list
	return np.array(result)  # list to ndarray


def early_stopping(log_value, best_value, stopping_step, flag_step=20):
	# early stopping strategy:

	if log_value >= best_value:
		stopping_step = 0
		best_value = log_value
	else:
		stopping_step += 1

	if stopping_step >= flag_step:
		print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
		should_stop = True
	else:
		should_stop = False
	return best_value, stopping_step, should_stop