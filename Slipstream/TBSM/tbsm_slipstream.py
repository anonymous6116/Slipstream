# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

### import packages ###
from __future__ import absolute_import, division, print_function, unicode_literals

# miscellaneous
import time
import os
from os import path
import random

# numpy and scikit-learn
import numpy as np
from sklearn.metrics import roc_auc_score

import pandas as pd 

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as Functional
from torch.nn.parameter import Parameter
#from torch.utils.tensorboard import SummaryWriter


#yass 
import os
import copy
import random
import torch.distributed as distributed
import multiprocessing 
from multiprocessing import Process, Pool, Manager, Queue, Lock, current_process
from multiprocessing import shared_memory
import math
import copy


# tbsm data
import tbsm_data_pytorch as tp

# set python, numpy and torch random seeds
def set_seed(seed, use_gpu):
		random.seed(seed)
		os.environ['PYTHONHASHSEED'] = str(seed)
		np.random.seed(seed)
		if use_gpu:
				torch.manual_seed(seed)
				torch.cuda.manual_seed(seed)
				torch.cuda.manual_seed_all(seed)   # if using multi-GPU.
				torch.backends.cudnn.benchmark = False
				torch.backends.cudnn.deterministic = True


### define time series layer (TSL) ###
class TSL_Net(nn.Module):
		def __init__(
						self,
						arch_interaction_op='dot',
						arch_attention_mechanism='mlp',
						ln=None,
						model_type="tsl",
						tsl_inner="def",
						mha_num_heads=8,
						ln_top=""
		):
				super(TSL_Net, self).__init__()

				# save arguments
				self.arch_interaction_op = arch_interaction_op
				self.arch_attention_mechanism = arch_attention_mechanism
				self.model_type = model_type
				self.tsl_inner = tsl_inner

				# setup for mechanism type
				if self.arch_attention_mechanism == 'mlp':
						self.mlp = dlrm.DLRM_Net().create_mlp(ln, len(ln) - 2)
						self.mlp = self.mlp.to("cuda:0")

				# setup extra parameters for some of the models
				if self.model_type == "tsl" and self.tsl_inner in ["def", "ind"]:
						m = ln_top[-1]  # dim of dlrm output
						mean = 0.0
						std_dev = np.sqrt(2 / (m + m))
						W = np.random.normal(mean, std_dev, size=(1, m, m)).astype(np.float32)
						self.A = Parameter(torch.tensor(W).to("cuda:0"), requires_grad=True)
				elif self.model_type == "mha":
						m = ln_top[-1]  # dlrm output dim
						self.nheads = mha_num_heads
						self.emb_m = self.nheads * m  # mha emb dim
						mean = 0.0
						std_dev = np.sqrt(2 / (m + m))  # np.sqrt(1 / m) # np.sqrt(1 / n)
						qm = np.random.normal(mean, std_dev, size=(1, m, self.emb_m)) \
								.astype(np.float32)
						self.Q = Parameter(torch.tensor(qm), requires_grad=True)
						km = np.random.normal(mean, std_dev, size=(1, m, self.emb_m))  \
								.astype(np.float32)
						self.K = Parameter(torch.tensor(km), requires_grad=True)
						vm = np.random.normal(mean, std_dev, size=(1, m, self.emb_m)) \
								.astype(np.float32)
						self.V = Parameter(torch.tensor(vm), requires_grad=True)

		def forward(self, x=None, H=None):
				# adjust input shape
				(batchSize, vector_dim) = x.shape
				x = torch.reshape(x, (batchSize, 1, -1))
				x = torch.transpose(x, 1, 2)
				# debug prints
				# print("shapes: ", self.A.shape, x.shape)

				# perform mode operation
				if self.model_type == "tsl":
						if self.tsl_inner == "def":
								ax = torch.matmul(self.A, x)
								x = torch.matmul(self.A.permute(0, 2, 1), ax)
								# debug prints
								# print("shapes: ", H.shape, ax.shape, x.shape)
						elif self.tsl_inner == "ind":
								x = torch.matmul(self.A, x)

						# perform interaction operation
						if self.arch_interaction_op == 'dot':
								if self.arch_attention_mechanism == 'mul':
										# coefficients
										a = torch.transpose(torch.bmm(H, x), 1, 2)
										# context
										c = torch.bmm(a, H)
								elif self.arch_attention_mechanism == 'mlp':
										# coefficients
										a = torch.transpose(torch.bmm(H, x), 1, 2)
										# MLP first/last layer dims are automatically adjusted to ts_length
										y = dlrm.DLRM_Net().apply_mlp(a, self.mlp)
										# context, y = mlp(a)
										c = torch.bmm(torch.reshape(y, (batchSize, 1, -1)), H)
								else:
										sys.exit('ERROR: --arch-attention-mechanism='
												+ self.arch_attention_mechanism + ' is not supported')

						else:
								sys.exit('ERROR: --arch-interaction-op=' + self.arch_interaction_op
										+ ' is not supported')

				elif self.model_type == "mha":
						x = torch.transpose(x, 1, 2)
						Qx = torch.transpose(torch.matmul(x, self.Q), 0, 1)
						HK = torch.transpose(torch.matmul(H, self.K), 0, 1)
						HV = torch.transpose(torch.matmul(H, self.V), 0, 1)
						# multi-head attention (mha)
						multihead_attn = nn.MultiheadAttention(self.emb_m, self.nheads).to(x.device)
						attn_output, _ = multihead_attn(Qx, HK, HV)
						# context
						c = torch.squeeze(attn_output, dim=0)
						# debug prints
						# print("shapes:", c.shape, Qx.shape)

				return c


### define Time-based Sequence Model (TBSM) ###
class TBSM_Net(nn.Module):
		def __init__(
						self,
						m_spa,
						ln_emb,
						ln_hot_emb,
						ln_bot,
						ln_top,
						arch_interaction_op,
						arch_interaction_itself,
						ln_mlp,
						ln_tsl,
						tsl_interaction_op,
						tsl_mechanism,
						ts_length,
						ndevices,
						model_type="",
						tsl_seq=False,
						tsl_proj=True,
						tsl_inner="def",
						tsl_num_heads=1,
						mha_num_heads=8,
						rnn_num_layers=5,
						debug_mode=False,
		):
				super(TBSM_Net, self).__init__()

				# save arguments
				self.ndevices = ndevices
				self.debug_mode = debug_mode
				self.ln_bot = ln_bot
				self.ln_top = ln_top
				self.ln_tsl = ln_tsl
				self.ts_length = ts_length
				self.tsl_interaction_op = tsl_interaction_op
				self.tsl_mechanism = tsl_mechanism
				self.model_type = model_type
				self.tsl_seq = tsl_seq
				self.tsl_proj = tsl_proj
				self.tsl_inner = tsl_inner
				self.tsl_num_heads = tsl_num_heads
				self.mha_num_heads = mha_num_heads
				self.rnn_num_layers = rnn_num_layers
				self.ams = nn.ModuleList()
				self.mlps = nn.ModuleList()
				if self.model_type == "tsl":
						self.num_mlps = int(self.tsl_num_heads)   # number of tsl components
				else:
						self.num_mlps = 1
				#debug prints
				if self.debug_mode:
						print(self.model_type)
						print(ln_bot)
						print(ln_top)
						print(ln_emb)
						print(ln_hot_emb)

				# embedding layer (implemented through dlrm tower, without last layer sigmoid)
				if "qr" in model_type:
						self.dlrm = dlrm.DLRM_Net(
								m_spa, ln_emb, ln_hot_emb, ln_bot, ln_top,
								arch_interaction_op, arch_interaction_itself,
								qr_flag=True, qr_operation="add", qr_collisions=4, qr_threshold=100000
						)
						print("Using QR embedding method.")
				else:
						self.dlrm = dlrm.DLRM_Net(
								m_spa, ln_emb, ln_hot_emb, ln_bot, ln_top,
								arch_interaction_op, arch_interaction_itself,
								ndevices=ndevices
						)
				print("DLRM MODEL CREATED")
				# prepare data needed for tsl layer construction
				if self.model_type == "tsl":
						if not self.tsl_seq:
								self.ts_array = [self.ts_length] * self.num_mlps
						else:
								self.ts_array = []
								m = self.ts_length / self.tsl_num_heads
								for j in range(self.tsl_num_heads, 0, -1):
										t = min(self.ts_length, round(m * j))
										self.ts_array.append(t)
				elif self.model_type == "mha":
						self.ts_array = [self.ts_length]
				else:
						self.ts_array = []

				# construction of one or more tsl components
				for ts in self.ts_array:

						ln_tsl = np.concatenate((np.array([ts]), self.ln_tsl))
						ln_tsl = np.append(ln_tsl, ts)

						# create tsl mechanism
						am = TSL_Net(
								arch_interaction_op=self.tsl_interaction_op,
								arch_attention_mechanism=self.tsl_mechanism,
								ln=ln_tsl, model_type=self.model_type,
								tsl_inner=self.tsl_inner,
								mha_num_heads=self.mha_num_heads, ln_top=self.ln_top,
						)

						self.ams.append(am)
				print("TSL MODEL CREATED")
				# tsl MLPs (with sigmoid on last layer)
				for _ in range(self.num_mlps):
						mlp_tsl = dlrm.DLRM_Net().create_mlp(ln_mlp, ln_mlp.size - 2)
						mlp_tsl = mlp_tsl.to("cuda:0")
						self.mlps.append(mlp_tsl)

				# top mlp if needed
				if self.num_mlps > 1:
						f_mlp = np.array([self.num_mlps, self.num_mlps + 4, 1])
						self.final_mlp = dlrm.DLRM_Net().create_mlp(f_mlp, f_mlp.size - 2)
						self.final_mlp = self.final_mlp.to("cuda:0")
				print("DONE")
		
		def forward(self, x, lS_o, lS_i, data):
				if data == "hot":
						return self.hot_forward(x, lS_o, lS_i, data)
				else:
						return self.normal_forward(x, lS_o, lS_i, data)

		def normal_forward(self, x, lS_o, lS_i, data):
				# data point is history H and last entry w
				n = x[0].shape[0]  # batch_size
				ts = len(x)
				#H = torch.zeros(n, self.ts_length, self.ln_top[-1]).to(x[0].device)
				H = torch.zeros(n, self.ts_length, self.ln_top[-1])
				# split point into first part (history)
				# and last item
				for j in range(ts - self.ts_length - 1, ts - 1):
						oj = j - (ts - self.ts_length - 1)
						v = self.dlrm(x[j], lS_o[j], lS_i[j], data, j)
						if self.model_type == "tsl" and self.tsl_proj:
								v = Functional.normalize(v, p=2, dim=1)
						H[:, oj, :] = v

				H = H.to("cuda:0")
				
				w = self.dlrm(x[-1], lS_o[-1], lS_i[-1], data, j)
				# project onto sphere
				if self.model_type == "tsl" and self.tsl_proj:
						w = Functional.normalize(w, p=2, dim=1)
				# print("data: ", x[-1], lS_o[-1], lS_i[-1])
				w = w.to("cuda:0")

				(mini_batch_size, _) = w.shape

				# for cases when model is tsl or mha
				if self.model_type != "rnn":

						# create MLP for each TSL component
						# each ams[] element is one component
						for j in range(self.num_mlps):

								ts = self.ts_length - self.ts_array[j]
								c = self.ams[j](w, H[:, ts:, :])
								c = torch.reshape(c, (mini_batch_size, -1))
								# concat context and w
								z = torch.cat([c, w], dim=1)
								# obtain probability of a click as a result of MLP
								p = dlrm.DLRM_Net().apply_mlp(z, self.mlps[j])
								if j == 0:
										ps = p
								else:
										ps = torch.cat((ps, p), dim=1)

						if ps.shape[1] > 1:
								p_out = dlrm.DLRM_Net().apply_mlp(ps, self.final_mlp)
						else:
								p_out = ps

				# RNN based on LSTM cells case, context is final hidden state
				else:
						hidden_dim = w.shape[1]     # equal to dim(w) = dim(c)
						level = self.rnn_num_layers  # num stacks of rnns
						Ht = H.permute(1, 0, 2)
						rnn = nn.LSTM(int(self.ln_top[-1]), int(hidden_dim),
						int(level)).to(x[0].device)
						h0 = torch.randn(level, n, hidden_dim).to(x[0].device)
						c0 = torch.randn(level, n, hidden_dim).to(x[0].device)
						output, (hn, cn) = rnn(Ht, (h0, c0))
						hn, cn = torch.squeeze(hn[level - 1, :, :]), \
								torch.squeeze(cn[level - 1, :, :])
						if self.debug_mode:
								print(w.shape, output.shape, hn.shape)
						# concat context and w
						z = torch.cat([hn, w], dim=1)
						p_out = dlrm.DLRM_Net().apply_mlp(z, self.mlps[0])

				return p_out

		def hot_forward(self, x, lS_o, lS_i, data):
				# data point is history H and last entry w
				n = x[0].shape[0]  # batch_size
				ts = len(x)
				H = torch.zeros(n, self.ts_length, self.ln_top[-1]).to(x[0].device)
				#H = torch.zeros(n, self.ts_length, self.ln_top[-1])
				# split point into first part (history)
				# and last item
				for j in range(ts - self.ts_length - 1, ts - 1):
						oj = j - (ts - self.ts_length - 1)
						v = self.dlrm(x[j], lS_o[j], lS_i[j], data, j)
						if self.model_type == "tsl" and self.tsl_proj:
								v = Functional.normalize(v, p=2, dim=1)
						H[:, oj, :] = v

				#H = H.to("cuda:0")
				
				w = self.dlrm(x[-1], lS_o[-1], lS_i[-1], data, j)
				# project onto sphere
				if self.model_type == "tsl" and self.tsl_proj:
						w = Functional.normalize(w, p=2, dim=1)
				# print("data: ", x[-1], lS_o[-1], lS_i[-1])
				#w = w.to("cuda:0")

				(mini_batch_size, _) = w.shape

				# for cases when model is tsl or mha
				if self.model_type != "rnn":

						# create MLP for each TSL component
						# each ams[] element is one component
						for j in range(self.num_mlps):

								ts = self.ts_length - self.ts_array[j]
								c = self.ams[j](w, H[:, ts:, :])
								c = torch.reshape(c, (mini_batch_size, -1))
								# concat context and w
								z = torch.cat([c, w], dim=1)
								# obtain probability of a click as a result of MLP
								p = dlrm.DLRM_Net().apply_mlp(z, self.mlps[j])
								if j == 0:
										ps = p
								else:
										ps = torch.cat((ps, p), dim=1)

						if ps.shape[1] > 1:
								p_out = dlrm.DLRM_Net().apply_mlp(ps, self.final_mlp)
						else:
								p_out = ps

				# RNN based on LSTM cells case, context is final hidden state
				else:
						hidden_dim = w.shape[1]     # equal to dim(w) = dim(c)
						level = self.rnn_num_layers  # num stacks of rnns
						Ht = H.permute(1, 0, 2)
						rnn = nn.LSTM(int(self.ln_top[-1]), int(hidden_dim),
						int(level)).to(x[0].device)
						h0 = torch.randn(level, n, hidden_dim).to(x[0].device)
						c0 = torch.randn(level, n, hidden_dim).to(x[0].device)
						output, (hn, cn) = rnn(Ht, (h0, c0))
						hn, cn = torch.squeeze(hn[level - 1, :, :]), \
								torch.squeeze(cn[level - 1, :, :])
						if self.debug_mode:
								print(w.shape, output.shape, hn.shape)
						# concat context and w
						z = torch.cat([hn, w], dim=1)
						p_out = dlrm.DLRM_Net().apply_mlp(z, self.mlps[0])

				return p_out



# construct tbsm model or read it from the file specified
# by args.save_model
def get_tbsm(args, use_gpu):

		# train, test, or train-test
		modes = args.mode.split("-")
		model_file = args.save_model

		if args.debug_mode:
				print("model_file: ", model_file)
				print("model_type: ", args.model_type)

		if use_gpu:
				ngpus = torch.cuda.device_count()  # 1
				devicenum = "cuda:" + str(args.device_num % ngpus)
				print("device:", devicenum)
				device = torch.device(devicenum)
				print("Using {} GPU(s)...".format(ngpus))
		else:
				device = torch.device("cpu")
				print("Using CPU...")

		# prepare dlrm arch
		m_spa = args.arch_sparse_feature_size
		print("m_spa : ", m_spa)
		# this is an array of sizes of cat features
		ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
		print("ln_emb : ", ln_emb)
		ln_hot_emb = args.arch_hot_embedding_size
		print("ln_hot_emb : ", ln_hot_emb)
		num_fea = ln_emb.size + 1  # user: num sparse + bot_mlp(all dense)
		ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
		print("ln_bot : ", ln_bot)
		#  m_den = ln_bot[0]
		ln_bot[ln_bot.size - 1] = m_spa  # enforcing
		m_den_out = ln_bot[ln_bot.size - 1]  # must be == m_spa (embed dim)

		if args.arch_interaction_op == "dot":
				# approach 1: all
				# num_int = num_fea * num_fea + m_den_out
				# approach 2: unique
				if args.arch_interaction_itself:
						num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
				else:
						num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
		elif args.arch_interaction_op == "cat":
				num_int = num_fea * m_den_out
		else:
				sys.exit(
						"ERROR: --arch-interaction-op="
						+ args.arch_interaction_op
						+ " is not supported"
				)
		arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
		ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")
		print("ln_top : ", ln_top)
		# sigmoid_top = len(ln_top) - 2    # used only if length_ts == 1
		# attention mlp (will be automatically adjusted so that first and last
		# layer correspond to number of vectors (ts_length) used in attention)
		ln_atn = np.fromstring(args.tsl_mlp, dtype=int, sep="-")
		print("ln_atn : ", ln_atn)
		# context MLP (with automatically adjusted first layer)
		if args.model_type == "mha":
				num_cat = (int(args.mha_num_heads) + 1) * ln_top[-1]    # mha with k heads + w
		else:         # tsl or rnn
				num_cat = 2 * ln_top[-1]   # [c,w]
		arch_mlp_adjusted = str(num_cat) + "-" + args.arch_mlp
		ln_mlp = np.fromstring(arch_mlp_adjusted, dtype=int, sep="-")
		print("ln_mlp : ", ln_mlp)
		ndevices = min(ngpus, args.mini_batch_size) if use_gpu else -1
		# construct TBSM
		tbsm = TBSM_Net(
				m_spa,
				ln_emb,
				ln_hot_emb,
				ln_bot,
				ln_top,
				args.arch_interaction_op,
				args.arch_interaction_itself,
				ln_mlp,
				ln_atn,
				args.tsl_interaction_op,
				args.tsl_mechanism,
				args.ts_length,
				ndevices,
				args.model_type,
				args.tsl_seq,
				args.tsl_proj,
				args.tsl_inner,
				args.tsl_num_heads,
				args.mha_num_heads,
				args.rnn_num_layers,
				args.debug_mode,
		)

		# move model to gpu
		if use_gpu:
				tbsm = tbsm
				

		return tbsm, device


def data_wrap(X, lS_o, lS_i, use_gpu, device, data):
		if use_gpu:  # .cuda()
				if data == "hot":
						return ([xj.to(device) for xj in X],
								[[S_o.to(device) for S_o in row] for row in lS_o],
								[[S_i.to(device) for S_i in row] for row in lS_i],
								data)
				else:
						return ([xj.to(device) for xj in X],
								[[S_o for S_o in row] for row in lS_o],
								[[S_i for S_i in row] for row in lS_i],
								data)
		else:
				return X, lS_o, lS_i, data


def time_wrap(use_gpu):
		if use_gpu:
				torch.cuda.synchronize()
		return time.time()


def loss_fn_wrap(Z, T, use_gpu, device):
		if use_gpu:
				return loss_fn(Z, T.to(device))
		else:
				return loss_fn(Z, T)

loss_fn = torch.nn.BCELoss(reduction="mean")

# iterate through validation data, which can be used to determine the best seed and
# during main training for deciding to save the current model
def iterate_val_data(val_ld, tbsm, use_gpu, device):
		# NOTE: call to tbsm.eval() not needed here, see
		# https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615
		total_loss_val = 0
		total_accu_test = 0
		total_samp_test = 0

		for _, (X, lS_o, lS_i, T_test) in enumerate(val_ld):
				data = "val"
				batchSize = X[0].shape[0]

				Z_test = tbsm(*data_wrap(X,
						lS_o,
						lS_i,
						use_gpu,
						device,
						data
				))

				# # compute loss and accuracy
				z = Z_test.detach().cpu().numpy()  # numpy array
				t = T_test.detach().cpu().numpy()  # numpy array
				A_test = np.sum((np.round(z, 0) == t).astype(np.uint8))
				total_accu_test += A_test
				total_samp_test += batchSize

				E_test = loss_fn_wrap(Z_test, T_test, use_gpu, device)
				L_test = E_test.detach().cpu().numpy()  # numpy array
				total_loss_val += (L_test * batchSize)

		return total_accu_test, total_samp_test, total_loss_val

# iterate through training data, which is called once every epoch. It updates weights,
# computes loss, accuracy, saves model if needed and calls iterate_val_data() function.
# isMainTraining is True for main training and False for fast seed selection
def iterate_train_data(args, train_hot_ld, train_normal_ld, hot_emb_dict, val_ld, tbsm, k, use_gpu, device, writer, losses, accuracies, isMainTraining, hot_entry_samples, ln_hot_emb, train_hot):
		# select number of batches
		print(" isMainTraining : ",isMainTraining)
		if isMainTraining:
				#nbatches = len(train_ld) if args.num_batches == 0 else args.num_batches
				nbatches_hot = args.num_batches if args.num_batches > 0 else len(train_hot_ld)
				nbatches_normal = args.num_batches if args.num_batches > 0 else len(train_normal_ld)

				ending_minibatch = round(args.minibatch_percentage * nbatches_hot) + args.starting_minibatch
				nbatches = nbatches_hot + nbatches_normal

				print(" ")
				print("#_________________________________________#")
				print("Minibatch_Size : ", args.mini_batch_size)
				print("Number_Hot_Minibatches : ", nbatches_hot)
				print("Number_Cold_Minibatches : ", nbatches_normal)
				print("Total_Minibatches : ", nbatches)
				print("Ending_Minibatch : ",ending_minibatch)
				print("Starting_Minibatch : ",args.starting_minibatch)
				print("#_________________________________________#")
				print(" ")
		else:
				nbatches = len(train_ld)




		# shared memory 
		shm_0 = shared_memory.SharedMemory(create=True, size=40 * 4 * len(train_hot))
		cluster_0 = np.ndarray(len(train_hot), dtype=object, buffer=shm_0.buf)

		shm_1 = shared_memory.SharedMemory(create=True, size=40 * 4 * len(train_hot))
		cluster_1 = np.ndarray(len(train_hot), dtype=object, buffer=shm_1.buf)




		# specify the optimizer algorithm
		optimizer = torch.optim.Adagrad(tbsm.parameters(), lr=args.learning_rate)

		total_time = 0
		total_loss = 0
		total_accu = 0
		total_iter = 0
		total_samp = 0
		max_gA_test = 0


		cold_forward_time = 0
		cold_backward_time = 0
		cold_optimizer_time = 0
		cold_scheduler_time =0
		


		hot_forward_time = 0
		hot_backward_time = 0
		hot_optimizer_time = 0
		hot_scheduler_time = 0
		

		cold_total = 0
		hot_total = 0

		forward_sampling_time = 0
		backward_sampling_time = 0
		optimizer_sampling_time = 0
		scheduler_sampling_time = 0


		cpu_operation_time = 0
		gpu_snapshot_time = 0
		gpu_operation_time = 0

		hot_emb_update = 0
		cold_emb_update = 0

		forward_time = 0
		backward_time = 0
		optimizer_time = 0

		test_time_cold = 0
		test_time_hot = 0



		accum_time_begin = time_wrap(use_gpu)
		
		# Using normal Train Data
		for j, (X, lS_o, lS_i, T) in enumerate(train_normal_ld):
				data = "normal"
				if j >= nbatches_normal - 2:
						break
				t1 = time_wrap(use_gpu)
				batchSize = X[0].shape[0]
				# forward pass
				begin_forward = time_wrap(use_gpu)

				
				Z = tbsm(*data_wrap(X,
						lS_o,
						lS_i,
						use_gpu,
						device,
						data
				))

				end_forward = time_wrap(use_gpu)

				# loss
				E = loss_fn_wrap(Z, T, use_gpu, device)
				# compute loss and accuracy
				L = E.detach().cpu().numpy()  # numpy array
				z = Z.detach().cpu().numpy()  # numpy array
				t = T.detach().cpu().numpy()  # numpy array
				# rounding t
				A = np.sum((np.round(z, 0) == np.round(t, 0)).astype(np.uint8))

				optimizer.zero_grad()

				# backward pass
				E.backward(retain_graph=True)

				end_backward = time_wrap(use_gpu)

				# weights update
				optimizer.step()

				end_optimizing = time_wrap(use_gpu)

				t2 = time_wrap(use_gpu)
				total_time += t2 - t1
				total_loss += (L * batchSize)
				total_accu += A
				total_iter += 1
				total_samp += batchSize

				cold_forward_time += end_forward - begin_forward
				cold_backward_time += end_backward - end_forward
				cold_optimizer_time += end_optimizing - end_backward

				cold_total += cold_forward_time 
				cold_total += cold_backward_time
				cold_total += cold_optimizer_time

				forward_time = end_forward - begin_forward
				backward_time = end_backward - end_forward
				optimizer_time = end_optimizing - end_backward


				print_tl = (j == 0) or ((j + 1) % args.print_freq == 0) or (j + 1 == nbatches_normal)
				# print time, loss and accuracy
				if print_tl and isMainTraining:

						gT = 1000.0 * total_time / total_iter if args.print_time else -1
						total_time = 0

						gL = total_loss / total_samp
						total_loss = 0

						gA = total_accu / total_samp
						total_accu = 0

						gForward = 1000 * forward_time / total_iter

						gBackward = 1000 * backward_time / total_iter

						gOptimizer = 1000 * optimizer_time / total_iter

						str_run_type = "inference" if args.inference_only else "training"
						
						print("Forward ", gForward)
						print("Backward ", gBackward)
						print("Optimizer ", gOptimizer)
						print("Epoch ", k)
						print("Iteration ", j + 1 )
						print("Total_Iterations ", nbatches)
						print("Iteration_time ", gT)
						print("Loss ", gL)
						print("Accuracy ", gA*100)
						print("Train_data ", data)
						print("\n")

						
						total_iter = 0
						total_samp = 0


				if isMainTraining:
						should_test = (
								(args.test_freq > 0
								and (j + 1) % args.test_freq == 0) or (j + 1 == nbatches_normal) or (j == 0)
						)
				else:
						should_test = (j == min(int(0.05 * len(train_ld)), len(train_ld) - 1))

				#  validation run

				#print(" isMainTraining ",isMainTraining)
				#print(" should_test ",should_test)
				
				if should_test:
				
						total_accu_test, total_samp_test, total_loss_val = iterate_val_data(val_ld, tbsm, use_gpu, device)

						gA_test = total_accu_test / total_samp_test
						if not isMainTraining:
								break

						gL_test = total_loss_val / total_samp_test


						if args.enable_summary and isMainTraining:
				
								losses = np.append(losses, np.array([[j, gL, gL_test]]),
								axis=0)
								accuracies = np.append(accuracies, np.array([[j, gA * 100,
								gA_test * 100]]), axis=0)

						
						# save model if best so far
						if gA_test > max_gA_test and isMainTraining:
								print("Saving current model...")
								max_gA_test = gA_test
								model_ = tbsm
								'''
								torch.save(
										{
												"model_state_dict": model_.state_dict(),
												# "opt_state_dict": optimizer.state_dict(),
										},
										args.save_model,
								)
								'''
						
						print("Test_Iteration ", j + 1)
						print("Total_Iterations ", nbatches)
						print("Test_Loss ", gL_test)
						print("Test_Accuracy ", gA_test * 100)
						print("Best_test_Accuracy ", max_gA_test * 100)
						print("\n")
				


		begin_emb_update = time_wrap(use_gpu)

		for _, emb_dict in enumerate(hot_emb_dict):
				for _, (emb_no, emb_row) in enumerate(emb_dict):
						hot_row = emb_dict[(emb_no, emb_row)]
						data = tbsm.dlrm.emb_l[emb_no].weight.data[emb_row]
						tbsm.dlrm.hot_emb_l[0].weight.data[hot_row] = data

		end_emb_update = time_wrap(use_gpu)
		cold_emb_update += (end_emb_update - begin_emb_update)
		print("\nEMB_Hot_Update :", cold_emb_update)
		print("\n") 

		




		# ================================ SETUP Sampling =========================================
		# Using Hot Train Data
		for j, (X, lS_o, lS_i, T) in enumerate(train_hot_ld):

				if j > ending_minibatch:
						break
				data = "hot"

				t1 = time_wrap(use_gpu)
				batchSize = X[0].shape[0]
				
				# forward pass
				begin_forward_sampling = time_wrap(use_gpu)

				Z = tbsm(*data_wrap(X,
						lS_o,
						lS_i,
						use_gpu,
						device,
						data
				))

				end_forward_sampling = time_wrap(use_gpu)



				# ============================= first snapshot taken ========================================
				if j == args.starting_minibatch and isMainTraining:
						begin = time_wrap(use_gpu)
						prev_emb_hot =  copy.deepcopy(tbsm.dlrm.hot_emb_l[0].weight)                 # making a copy on GPU - snapshop of the whole hot embedding 
						end = time_wrap(use_gpu)
						gpu_operation_time += (end - begin)
				# ===========================================================================================


				# loss
				E = loss_fn_wrap(Z, T, use_gpu, device)
				# compute loss and accuracy
				L = E.detach().cpu().numpy()  # numpy array
				z = Z.detach().cpu().numpy()  # numpy array
				t = T.detach().cpu().numpy()  # numpy array
				# rounding t
				A = np.sum((np.round(z, 0) == np.round(t, 0)).astype(np.uint8))

				optimizer.zero_grad()

				# backward pass
				E.backward(retain_graph=True)

				end_backward_sampling = time_wrap(use_gpu)

				# weights update
				optimizer.step()

				end_optimizing_sampling = time_wrap(use_gpu)

				t2 = time_wrap(use_gpu)



				# ======================== Second Screenshot ===============================
				if (k == 0 and j == ending_minibatch and isMainTraining):
						begin = time_wrap(use_gpu)
						after_emb_hot =  copy.deepcopy(tbsm.dlrm.hot_emb_l[0].weight) 
						end = time_wrap(use_gpu)
						gpu_operation_time += (end - begin)
				# ==========================================================================



				total_time += t2 - t1
				total_loss += (L * batchSize)
				total_accu += A
				total_iter += 1
				total_samp += batchSize

				forward_sampling_time += end_forward_sampling - begin_forward_sampling
				backward_sampling_time += end_backward_sampling - end_forward_sampling
				backward_sampling_time += end_optimizing_sampling - end_backward_sampling


				forward_time = end_forward - begin_forward
				backward_time = end_backward - end_forward
				optimizer_time = end_optimizing - end_backward


				print_tl = (j == 0) or ((j + 1+ nbatches_normal) % args.print_freq == 0) or (j + 1 == nbatches_hot)
				# print time, loss and accuracy
				if print_tl and isMainTraining:

						gT = 1000.0 * total_time / total_iter if args.print_time else -1
						total_time = 0

						gL = total_loss / total_samp
						total_loss = 0

						gA = total_accu / total_samp
						total_accu = 0

						gForward = 1000 * forward_time / total_iter

						gBackward = 1000 * backward_time / total_iter

						gOptimizer = 1000 * optimizer_time / total_iter

						str_run_type = "inference" if args.inference_only else "training"
						
						print("Forward ", gForward)
						print("Backward ", gBackward)
						print("Optimizer ", gOptimizer)
						print("Epoch ", k)
						print("Iteration ", j + 1 + nbatches_normal)
						print("Total_Iterations ", nbatches)
						print("Iteration_time ", gT)
						print("Loss ", gL)
						print("Accuracy ", gA*100)
						print("Train_data ", data)
						print("\n")

						total_iter = 0
						total_samp = 0

				if isMainTraining:
						should_test = (
								(args.test_freq > 0
								and (j + 1 + nbatches_normal) % args.test_freq == 0) or (j + 1 == nbatches_hot) or (j == 0)
						)
				else:
						should_test = (j == min(int(0.05 * len(train_ld)), len(train_ld) - 1))

				#  validation run
				
				if should_test:
				
						# Before testing update the emb_l using hot_emb_l
						# ======================= Updating the emb_l with hot_emb_l =====================
												
						begin_emb_update = time_wrap(use_gpu)

						hot_emb = tbsm.dlrm.hot_emb_l[0].weight.detach().cpu().numpy()
												
						for _, emb_dict in enumerate(hot_emb_dict):
								for _, (emb_no, emb_row) in enumerate(emb_dict):
										hot_row = emb_dict[(emb_no, emb_row)]
										data = torch.tensor(hot_emb[hot_row])
										tbsm.dlrm.emb_l[emb_no].weight.data[emb_row] = data

						end_emb_update = time_wrap(use_gpu)
						hot_emb_update += (end_emb_update - begin_emb_update)
						print("\nEMB_normal_Update ", cold_emb_update)
						print("\n")
						
												
						# ===============================================================================

						total_accu_test, total_samp_test, total_loss_val = iterate_val_data(val_ld, tbsm, use_gpu, device)

						gA_test = total_accu_test / total_samp_test
						if not isMainTraining:
								break

						gL_test = total_loss_val / total_samp_test


						if args.enable_summary and isMainTraining:
						
								losses = np.append(losses, np.array([[j, gL, gL_test]]),
								axis=0)
								accuracies = np.append(accuracies, np.array([[j, gA * 100,
								gA_test * 100]]), axis=0)

						# save model if best so far
						'''
						if gA_test > max_gA_test and isMainTraining:
								print("Saving current model...")
								max_gA_test = gA_test
								model_ = tbsm
								torch.save(
										{
												"model_state_dict": model_.state_dict(),
												# "opt_state_dict": optimizer.state_dict(),
										},
										args.save_model,
								)
						'''
						print("Test_Iteration ", j + 1 + nbatches_normal)
						print("Total_Iterations ", nbatches)
						print("Test_Loss ", gL_test)
						print("Test_Accuracy ", gA_test * 100)
						print("Best_test_Accuracy ", max_gA_test * 100)
						print("\n")

		# At the end of train_hot_ld last iteration update the emb_l
		# ======================= Updating the emb_l with hot_emb_l =====================
												
		begin_emb_update = time_wrap(use_gpu)
		hot_emb = tbsm.dlrm.hot_emb_l[0].weight.detach().cpu().numpy()
												
		for _, emb_dict in enumerate(hot_emb_dict):
				for _, (emb_no, emb_row) in enumerate(emb_dict):
						hot_row = emb_dict[(emb_no, emb_row)]
						data = torch.tensor(hot_emb[hot_row])
						tbsm.dlrm.emb_l[emb_no].weight.data[emb_row] = data

		end_emb_update = time_wrap(use_gpu)
		cold_emb_update += (end_emb_update - begin_emb_update)
		print("\nEMB_normal_Update ", cold_emb_update)
		print("\n")
											
		# ===============================================================================
		








		# =================================== Finding THE Threshold !! ========================================= #
		lower_drop_percentage = args.target_drop_percentage - 0.02					
		upper_drop_percentage = args.target_drop_percentage + 0.02

		# drop % is at minibatch level
		hot_entry_samples = int(args.sample_rate * len(train_hot))
		print("Number_Samples_Threshold_Setting : ",hot_entry_samples)
		sampled_train_data = np.random.randint(args.mini_batch_size * ending_minibatch, len(train_hot), size = hot_entry_samples )
		random_hot_emb = []
		
		#for i, idx in enumerate(sampled_train_data):
		#	random_hot_emb.append(train_hot[idx])

		# more optimized ?
		random_hot_emb = [train_hot[idx] for i, idx in enumerate(sampled_train_data)]

		input_counter = 0
		total_threshold_setting_time = 0
		threshold_upper = 0.1 
		threshold_lower = 0
		final_threshold = 0
		num_cores = multiprocessing.cpu_count()

		# Using Hot Train Data
		for minibatch in range(0,25):
			not_changing_inputs = 0
			begin_threshold = time_wrap(use_gpu)

			
			result_total = abs(prev_emb_hot - after_emb_hot) < (threshold_upper + threshold_lower)/2 
			final_total = torch.all(result_total, dim=1)
			final_total = final_total.int()
			final_total_number = torch.sum(final_total)
			final_total = final_total.detach().cpu().numpy() 
			
			for first, hot_tuple in enumerate(random_hot_emb):
				input_counter = 0
				
				lS_i_temp = []
				#print(hot_tuple)
				
				for second, lS_i_row in enumerate(hot_tuple[0]):
					temp_list = []
					#print("lS_i_row : ",lS_i_row)
					# whole row of 21 access to the 3 embs
					for third, lS_i_index in enumerate(lS_i_row):
						# it's the int
						if final_total[int(lS_i_index)] == 1:
							#input_counter = input_counter + 1
							temp_list.append(int(lS_i_index))
					

					#print("len(temp_list) : ",len(temp_list))
					if len(temp_list) >= args.non_changing_index:
						lS_i_temp.append(temp_list)


				# all 3 inputs should be above threshold
				if (len(lS_i_temp) == len(hot_tuple[0]) ):
					not_changing_inputs = not_changing_inputs + 1

			
			#print("not_changing_inputs ",not_changing_inputs)
			ratio = not_changing_inputs/hot_entry_samples
			
			if lower_drop_percentage <= ratio and  ratio <= upper_drop_percentage:
				print("Final Threshold : ", (threshold_upper + threshold_lower)/2)
				print("Ratio : ", ratio)
				final_threshold = (threshold_upper + threshold_lower)/2
				break

			# we should set the threshold higher to drop more 
			elif lower_drop_percentage > ratio:
				threshold_lower = (threshold_upper + threshold_lower)/2


			# setting the threshold lower to drop less
			elif ratio >= upper_drop_percentage:
				threshold_upper = (threshold_upper + threshold_lower)/2

			else:
				print(" why are we here? !!!!!")

			end_threshold = time_wrap(use_gpu) 
			total_threshold_setting_time += (end_threshold - begin_threshold)

		# =========================  DONE with finding thresholds ====================== #


		print("Final_Threshold : ", final_threshold)

		num_cores = multiprocessing.cpu_count()
		begin_cpu_operation = time_wrap(use_gpu)

		# clipping the inputs we ran already
		train_hot_2 = train_hot[args.mini_batch_size * ending_minibatch: len(train_hot)]
		chunksize = len(train_hot_2)//(num_cores)
		
 
		result_total = abs(prev_emb_hot - after_emb_hot) < final_threshold             # result for overall embedding table
		final_total = torch.all(result_total, dim=1)
		final_total = final_total.int()
		final_total_number = torch.sum(final_total)
		final_total = final_total.detach().cpu().numpy() 


		processes = [Process(target = worker_func,
							name = "%i" % i,
							args = (train_hot_2[i*chunksize : len(train_hot) if i == (num_cores-1) else (i+1)*chunksize],
									args.non_changing_index,
									final_total,
									cluster_0,
									cluster_1,
									chunksize
									)
							)
					
					for i in range(0, num_cores)]
		


		for process in processes:
			process.start()
		
		for process in processes:
			process.join()
		
		print("Number_of_Processes : ", len(processes))
			

		# removing "None" from the cluster 1 & 0
		nan_array = pd.isnull(cluster_1)
		not_nan_array = ~ nan_array
		cluster_1 = cluster_1[not_nan_array]

		nan_array = pd.isnull(cluster_0)
		not_nan_array = ~ nan_array
		cluster_0 = cluster_0[not_nan_array]


		end_cpu_operation = time_wrap(use_gpu)
		cpu_operation_time += end_cpu_operation - begin_cpu_operation
		print("Overall_CPU_Time_Accelerated : ", end_cpu_operation )
		print("Overall_Threshold_Quest_Time : ",total_threshold_setting_time)

		print(" ")
		print("==================================================")
		print("Number_of_Elements_Changing_Cluster0 : ",len(cluster_0))
		print("Number_of_Elements_Non_Changing_Cluster1 : ",len(cluster_1))
		print("Drop Percentage of Hot Inputs : ",len(cluster_1)/len(train_hot) *100,"%")
		print("==================================================")
		print(" ")

		cluster_0 = np.array(cluster_0).astype(object)      # non-skippable hot entries -- forming new hot minibatches
		cluster_1 = np.array(cluster_1).astype(object)


		train_hot_ld_cluster_0 = tp.alibaba_load_preprocessed_data_and_loaders(args, cluster_0)
		train_hot_ld_cluster_1 = tp.alibaba_load_preprocessed_data_and_loaders(args, cluster_1)

		nbatches_hot_cluster_0 = args.num_batches if args.num_batches > 0 else len(train_hot_ld_cluster_0)
		nbatches_hot_cluster_1 = args.num_batches if args.num_batches > 0 else len(train_hot_ld_cluster_1)

		shm_0.close()   
		shm_1.close()
		shm_0.unlink() 
		shm_1.unlink()

		#================================= DONE WITH SlipStream Threshold ================================ #






		# ==============================================================================================
		# Using Hot Train Data
		for j, (X, lS_o, lS_i, T) in enumerate(train_hot_ld_cluster_0):

				data = "hot"

				if j >= nbatches_hot - 2:
						break
				t1 = time_wrap(use_gpu)
				batchSize = X[0].shape[0]
				
				# forward pass
				begin_forward_hot = time_wrap(use_gpu)

				Z = tbsm(*data_wrap(X,
						lS_o,
						lS_i,
						use_gpu,
						device,
						data
				))

				end_forward_hot = time_wrap(use_gpu)

				# loss
				E = loss_fn_wrap(Z, T, use_gpu, device)
				# compute loss and accuracy
				L = E.detach().cpu().numpy()  # numpy array
				z = Z.detach().cpu().numpy()  # numpy array
				t = T.detach().cpu().numpy()  # numpy array
				# rounding t
				A = np.sum((np.round(z, 0) == np.round(t, 0)).astype(np.uint8))

				optimizer.zero_grad()

				# backward pass
				E.backward(retain_graph=True)

				end_backward_hot = time_wrap(use_gpu)

				# weights update
				optimizer.step()

				end_optimizing_hot = time_wrap(use_gpu)

				t2 = time_wrap(use_gpu)
				total_time += t2 - t1
				total_loss += (L * batchSize)
				total_accu += A
				total_iter += 1
				total_samp += batchSize

				hot_forward_time += end_forward_hot - begin_forward_hot
				hot_backward_time += end_backward_hot - end_forward_hot
				hot_optimizer_time += end_optimizing_hot - end_backward_hot

				print_tl = (j == 0) or ((j + 1 + nbatches_normal + ending_minibatch) % args.print_freq == 0) or (j + 1 == nbatches_hot)
				# print time, loss and accuracy
				if print_tl and isMainTraining:

						gT = 1000.0 * total_time / total_iter if args.print_time else -1
						total_time = 0

						gL = total_loss / total_samp
						total_loss = 0

						gA = total_accu / total_samp
						total_accu = 0

						gForward = 1000 * forward_time / total_iter

						gBackward = 1000 * backward_time / total_iter

						gOptimizer = 1000 * optimizer_time / total_iter

						str_run_type = "inference" if args.inference_only else "training"
						
						print("Forward ", gForward)
						print("Backward ", gBackward)
						print("Optimizer ", gOptimizer)
						print("Epoch ", k)
						print("Iteration ", j + 1 + nbatches_normal + ending_minibatch )
						print("Total_Iterations ", nbatches)
						print("Iteration_time ", gT)
						print("Loss ", gL)
						print("Accuracy ", gA*100)
						print("Train_data ", data)
						print("\n")

						total_iter = 0
						total_samp = 0
						forward_time = 0
						backward_time = 0
						optimizer_time = 0

				if isMainTraining:
						should_test = (
								(args.test_freq > 0
								and (j + 1 + nbatches_normal + ending_minibatch) % args.test_freq == 0) or (j + 1 == nbatches_hot) or (j == 0)
						)
				else:
						should_test = (j == min(int(0.05 * len(train_ld)), len(train_ld) - 1))

				#  validation run
				if should_test:

						# Before testing update the emb_l using hot_emb_l
						# ======================= Updating the emb_l with hot_emb_l =====================
												
						begin_emb_update = time_wrap(use_gpu)

						hot_emb = tbsm.dlrm.hot_emb_l[0].weight.detach().cpu().numpy()
												
						for _, emb_dict in enumerate(hot_emb_dict):
								for _, (emb_no, emb_row) in enumerate(emb_dict):
										hot_row = emb_dict[(emb_no, emb_row)]
										data = torch.tensor(hot_emb[hot_row])
										tbsm.dlrm.emb_l[emb_no].weight.data[emb_row] = data

						end_emb_update = time_wrap(use_gpu)

						print("\nEMB_normal_Update ", (end_emb_update - begin_emb_update))
						print("\n")
						
												
						# ===============================================================================

						total_accu_test, total_samp_test, total_loss_val = iterate_val_data(val_ld, tbsm, use_gpu, device)

						gA_test = total_accu_test / total_samp_test
						if not isMainTraining:
								break

						gL_test = total_loss_val / total_samp_test


						if args.enable_summary and isMainTraining:
						
								losses = np.append(losses, np.array([[j, gL, gL_test]]),
								axis=0)
								accuracies = np.append(accuracies, np.array([[j, gA * 100,
								gA_test * 100]]), axis=0)

						# save model if best so far
						if gA_test > max_gA_test and isMainTraining:
								print("Saving current model...")
								max_gA_test = gA_test
								model_ = tbsm
								'''
								torch.save(
										{
												"model_state_dict": model_.state_dict(),
												# "opt_state_dict": optimizer.state_dict(),
										},
										args.save_model,
								)
								'''

						print("Test_Iteration ", j + 1 +  nbatches_normal + ending_minibatch)
						print("Total_Iterations ", nbatches)
						print("Test_Loss ", gL_test)
						print("Test_Accuracy ", gA_test * 100)
						print("Best_test_Accuracy ", max_gA_test * 100)
						print("\n")

		# At the end of train_hot_ld last iteration update the emb_l
		# ======================= Updating the emb_l with hot_emb_l =====================
												
		begin_emb_update = time_wrap(use_gpu)

		hot_emb = tbsm.dlrm.hot_emb_l[0].weight.detach().cpu().numpy()
												
		for _, emb_dict in enumerate(hot_emb_dict):
				for _, (emb_no, emb_row) in enumerate(emb_dict):
						hot_row = emb_dict[(emb_no, emb_row)]
						data = torch.tensor(hot_emb[hot_row])
						tbsm.dlrm.emb_l[emb_no].weight.data[emb_row] = data

		end_emb_update = time_wrap(use_gpu)

		print("\nEMB_normal_Update ", (end_emb_update - begin_emb_update))
		print("\n")
		accum_time_end = time_wrap(use_gpu)








		print("=====================================================================")
		print("Epoch :", k)
		print("Total_Execution_Time_Sec :", (accum_time_end - accum_time_begin))
		print("=====================================================================")
		print(" ")

		print("=====================================================================")
		print("Cold_Forward_Time_Sec :", cold_forward_time)
		print("Cold_Backward_Time_Sec :", cold_backward_time)
		print("Cold_Optimizer_Time_Sec :", cold_optimizer_time)
		print("Cold_Scheduler_Time_Sec :", cold_scheduler_time)

		cold_overall_training = cold_forward_time + cold_backward_time + cold_optimizer_time + cold_scheduler_time
		print("Overall_Cold_Training_Time_Sec :",cold_overall_training)
		print("Testing_Time_Cold_Sec :",test_time_cold)
		print("Overall_Testing_Training_Time_Cold_Sec :",test_time_cold + cold_overall_training)
		print("=====================================================================")
		print(" ")
		
		print("=====================================================================")
		print("Sampling_Forward_Time_Sec :",forward_sampling_time)
		print("Sampling_Backward_Time_Sec :",backward_sampling_time)
		print("Sampling_Optimizer_Time_Sec :",optimizer_sampling_time)
		print("Sampling_Sch_Time_Sec :",scheduler_sampling_time)


		print(" ")
		print("GPU_Operation_Time_Sec :", gpu_operation_time)
		print("CPU_Operation_Time_Sec :", cpu_operation_time)

		slipstream_overall_overhead = gpu_operation_time + cpu_operation_time  + total_threshold_setting_time

		print("Overall_CPU_Time_Accelerated : ", cpu_operation_time )
		print("Overall_Threshold_Quest_Time : ",total_threshold_setting_time)
		print("Overall_SlipStream_Time_Sec :", slipstream_overall_overhead)
		print("=====================================================================")
		print(" ")
		
		print("=====================================================================")
		print("Hot_Forward_Time_Sec :", hot_forward_time)
		print("Hot_Backward_Time_Sec :", hot_backward_time)
		print("Hot_Optimizer_Time_Sec :", hot_optimizer_time)
		print("Hot_Scheduler_Time_Sec :", hot_scheduler_time)
		sampling_total = forward_sampling_time + backward_sampling_time + optimizer_sampling_time + scheduler_sampling_time 
		hot_total = hot_forward_time + hot_backward_time + hot_optimizer_time + hot_scheduler_time

		hot_overall_training = hot_total + sampling_total
		print("Overall_Hot_Training_Time_Sec :",hot_overall_training)
		print(" ")
		print("Testing_Time_Hot_Sec :",test_time_hot)
		print("Overall_Testing_Training_Time_Hot_Sec :",test_time_hot + hot_overall_training)
		print("=====================================================================")
		print(" ")


		print("=====================================================================")
		overall_emb_update = cold_emb_update + hot_emb_update
		print("Normal_EMB_Update : ", cold_emb_update)
		print("Hot_EMB_Update : ",hot_emb_update)
		print("Overall_EMB_Update : ", cold_emb_update + hot_emb_update)
		print("=====================================================================")
		print(" ")

		overall_training_time =  cold_overall_training + hot_overall_training + overall_emb_update + slipstream_overall_overhead
		print("=====================================================================")
		print("Overall_Training_Time_Sec : ",overall_training_time)
		print("Overall_Training_Testing_Time_Sec : ", test_time_hot + test_time_cold + overall_training_time)
		print("Overall_Threshold_Quest_Time_Percentage : ",slipstream_overall_overhead/overall_training_time*100)
		print("=====================================================================")
		print(" ")


		if not isMainTraining:
				return gA_test



	




# selects best seed, and does main model training
def train_tbsm(args, use_gpu):
		# prepare the data
		#train_ld, _ = tp.make_tbsm_data_and_loader(args, "train")
		val_ld, _ = tp.make_tbsm_data_and_loader(args, "val")

		# ============================== Loading processed hot data and normal data =======================
		print("Loading pre-processed Data")

		train_normal = np.load(args.train_normal_file, allow_pickle = True)
		train_normal = train_normal['arr_0']
		train_normal = train_normal.tolist()
		print("Length Train normal : ", len(train_normal))

		train_hot = np.load(args.train_hot_file, allow_pickle = True)
		train_hot = train_hot['arr_0']
		train_hot = train_hot.tolist()
		print("Length Train hot : ", len(train_hot))


		hot_emb_dict = np.load(args.hot_emb_dict_file, allow_pickle = True)
		hot_emb_dict = hot_emb_dict['arr_0']
		hot_emb_dict = hot_emb_dict.tolist()
		print("Length Hot emb Dict : ", len(hot_emb_dict))
		ln_hot_emb = 0
		for i, dict in enumerate(hot_emb_dict):
				ln_hot_emb = ln_hot_emb + len(dict)
		print("ln_hot_emb : ", ln_hot_emb)

		args.arch_hot_embedding_size = ln_hot_emb

		hot_entry_samples = int(args.sample_rate * ln_hot_emb)
		train_hot_ld, train_normal_ld = tp.load_preprocessed_data_and_loaders(args, train_hot, train_normal)
		# ==================================================================================================    

		# setup initial values
		isMainTraining = False
		#writer = SummaryWriter()
		writer = 0
		losses = np.empty((0,3), np.float32)
		accuracies = np.empty((0,3), np.float32)

		# selects best seed out of 5. Sometimes Adagrad gets stuck early, this
		# seems to occur randomly depending on initial weight values and
		# is independent of chosen model: N-inner, dot etc.
		# this procedure is used to reduce the probability of this happening.
		def select(args):

				seeds = np.random.randint(2, 10000, size=5)
				if args.debug_mode:
						print(seeds)
				best_index = 0
				max_val_accuracy = 0.0
				testpoint = min(int(0.05 * len(train_ld)), len(train_ld) - 1)
				print("testpoint, total batches: ", testpoint, len(train_ld))

				for i, seed in enumerate(seeds):

						set_seed(seed, use_gpu)
						tbsm, device = get_tbsm(args, use_gpu)

						gA_test = iterate_train_data(args, train_ld, val_ld, tbsm, 0, use_gpu,
																				 device, writer, losses, accuracies,
																				 isMainTraining, hot_entry_samples, ln_hot_emb, train_hot)

						if args.debug_mode:
								print("select: ", i, seed, gA_test, max_val_accuracy)
						if gA_test > max_val_accuracy:
								best_index = i
								max_val_accuracy = gA_test

				return seeds[best_index]

		# select best seed if needed
		seed = args.numpy_rand_seed
		
		# create or load TBSM
		tbsm, device = get_tbsm(args, use_gpu)
		if args.debug_mode:
				print("initial parameters (weights and bias):")
				for name, param in tbsm.named_parameters():
						print(name)
						print(param.detach().cpu().numpy())

		# main training loop
		isMainTraining = True
		print("time/loss/accuracy (if enabled):")
		#with torch.autograd.profiler.profile(args.enable_profiling, use_gpu) as prof:
		for k in range(args.nepochs):
				iterate_train_data(args, train_hot_ld, train_normal_ld, hot_emb_dict, val_ld, tbsm, k, use_gpu, device,
				writer, losses, accuracies, isMainTraining,hot_entry_samples , ln_hot_emb, train_hot)

		# debug prints
		if args.debug_mode:
				print("final parameters (weights and bias):")
				for name, param in tbsm.named_parameters():
						print(name)
						print(param.detach().cpu().numpy())

		# profiling
		if args.enable_profiling:
				with open("tbsm_pytorch.prof", "w") as prof_f:
						prof_f.write(
								prof.key_averages(group_by_input_shape=True).table(
										sort_by="self_cpu_time_total"
								)
						)
						prof.export_chrome_trace("./tbsm_pytorch.json")

		return




def worker_func(train_hot,non_changing_index,final_total, cluster_0, cluster_1, chunksize):
	clstr_0 = 0
	clstr_1 = 0
	index = int(current_process().name)
	
	for i, hot_tuple in enumerate(train_hot):
		lS_i_temp = []
		for j, lS_i_row in enumerate(hot_tuple[0]):
			temp_list = []
			for k, lS_i_index in enumerate(lS_i_row):
				if final_total[int(lS_i_index)] == 1:
					# for each input 
					temp_list.append(int(lS_i_index))


			if len(temp_list) >= non_changing_index:
				lS_i_temp.append(temp_list)

		if (len(lS_i_temp) == len(hot_tuple[0]) ):
			# all inputs are skipable and not changing 
			cluster_1[index*chunksize + clstr_1] = hot_tuple         
			clstr_1 = clstr_1 + 1
		else:
			cluster_0[index*chunksize + clstr_0] = hot_tuple
			clstr_0 = clstr_0 + 1





# evaluates model on test data and computes AUC metric
def test_tbsm(args, use_gpu):
		# prepare data
		test_ld, N_test = tp.make_tbsm_data_and_loader(args, "test")

		# setup initial values
		z_test = np.zeros((N_test, ), dtype=np.float)
		t_test = np.zeros((N_test, ), dtype=np.float)

		# check saved model exists
		if not path.exists(args.save_model):
				sys.exit("Can't find saved model. Exiting...")

		# create or load TBSM
		tbsm, device = get_tbsm(args, use_gpu)
		print(args.save_model)

		# main eval loop
		# NOTE: call to tbsm.eval() not needed here, see
		# https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615
		offset = 0
		for _, (X, lS_o, lS_i, T) in enumerate(test_ld):

				batchSize = X[0].shape[0]

				Z = tbsm(*data_wrap(X,
						lS_o,
						lS_i,
						use_gpu,
						device
				))

				z_test[offset: offset + batchSize] = np.squeeze(Z.detach().cpu().numpy(),
				axis=1)
				t_test[offset: offset + batchSize] = np.squeeze(T.detach().cpu().numpy(),
				axis=1)
				offset += batchSize

		if args.quality_metric == "auc":
				# compute AUC metric
				auc_score = 100.0 * roc_auc_score(t_test.astype(int), z_test)
				print("auc score: ", auc_score)
		else:
				sys.exit("Metric not supported.")


if __name__ == "__main__":
		### import packages ###

		import sys
		import argparse

		### parse arguments ###
		parser = argparse.ArgumentParser(description="Time Based Sequence Model (TBSM)")
		# path to dlrm
		parser.add_argument("--dlrm-path", type=str, default="")
		# data type: taobao or synthetic (generic)
		parser.add_argument("--datatype", type=str, default="synthetic")
		# mode: train or inference or both
		parser.add_argument("--mode", type=str, default="train")   # train, test, train-test
		# data locations
		parser.add_argument("--raw-train-file", type=str, default="./input/train.txt")
		parser.add_argument("--pro-train-file", type=str, default="./output/train.npz")
		parser.add_argument("--raw-test-file", type=str, default="./input/test.txt")
		parser.add_argument("--pro-test-file", type=str, default="./output/test.npz")
		parser.add_argument("--pro-val-file", type=str, default="./output/val.npz")
		parser.add_argument("--num-train-pts", type=int, default=100)
		parser.add_argument("--num-val-pts", type=int, default=20)
		# ========================= Added train files and dict ==============================
		parser.add_argument("--train-hot-file", type=str, default="") # train_hot.npz
		parser.add_argument("--train-normal-file", type=str, default="") # train_normal.npz
		parser.add_argument("--hot-emb-dict-file", type=str, default="") # hot_emb_dict.npz
		# ===================================================================================
		# time series length for train/val and test
		parser.add_argument("--ts-length", type=int, default=20)
		# model_type = "tsl", "mha", "rnn"
		parser.add_argument("--model-type", type=str, default="tsl")  # tsl, mha, rnn
		parser.add_argument("--tsl-seq", action="store_true", default=False)  # k-seq method
		parser.add_argument("--tsl-proj", action="store_true", default=True)  # sphere proj
		parser.add_argument("--tsl-inner", type=str, default="def")   # ind, def, dot
		parser.add_argument("--tsl-num-heads", type=int, default=1)   # num tsl components
		parser.add_argument("--mha-num-heads", type=int, default=8)   # num mha heads
		parser.add_argument("--rnn-num-layers", type=int, default=5)  # num rnn layers
		# num positive (and negative) points per user
		parser.add_argument("--points-per-user", type=int, default=10)
		# model arch related parameters
		# embedding dim for all sparse features (same for all features)
		parser.add_argument("--arch-sparse-feature-size", type=int, default=4)  # emb_dim
		# number of distinct values for each sparse feature
		parser.add_argument("--arch-embedding-size", type=str, default="4-3-2")  # vectors
		# number of distinct values hot feature
		parser.add_argument("--arch-hot-embedding-size", type=int, default=0)  # vectors
		# for taobao use "987994-4162024-9439")
		# MLP 1: num dense fea --> embedding dim for sparse fea (out_dim enforced)
		parser.add_argument("--arch-mlp-bot", type=str, default="1-4")
		# MLP 2: num_interactions + bot[-1] --> top[-1]
		# (in_dim adjusted, out_dim can be any)
		parser.add_argument("--arch-mlp-top", type=str, default="2-2")
		# MLP 3: attention: ts_length --> ts_length (both adjusted)
		parser.add_argument("--tsl-mlp", type=str, default="2-2")
		# MLP 4: final prob. of click: 2 * top[-1] --> [0,1] (in_dim adjusted)
		parser.add_argument("--arch-mlp", type=str, default="4-1")
		# interactions
		parser.add_argument("--arch-interaction-op", type=str, default="dot")
		parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
		parser.add_argument("--tsl-interaction-op", type=str, default="dot")
		parser.add_argument("--tsl-mechanism", type=str, default="mlp")  # mul or MLP
		# data
		parser.add_argument("--num-batches", type=int, default=0)
		# training
		parser.add_argument("--mini-batch-size", type=int, default=1)
		parser.add_argument("--nepochs", type=int, default=1)
		parser.add_argument("--learning-rate", type=float, default=0.05)
		parser.add_argument("--print-precision", type=int, default=5)
		parser.add_argument("--numpy-rand-seed", type=int, default=123)
		parser.add_argument("--no-select-seed", action="store_true", default=False)
		# inference
		parser.add_argument("--quality-metric", type=str, default="auc")
		parser.add_argument("--test-freq", type=int, default=0)
		parser.add_argument("--inference-only", type=bool, default=False)
		# saving model
		parser.add_argument("--save-model", type=str, default="./output/model.pt")
		# gpu
		parser.add_argument("--use-gpu", action="store_true", default=False)
		parser.add_argument("--device-num", type=int, default=0)
		# debugging and profiling
		parser.add_argument("--debug-mode", action="store_true", default=False)
		parser.add_argument("--print-freq", type=int, default=1)
		parser.add_argument("--print-time", action="store_true", default=False)
		parser.add_argument("--enable-summary", action="store_true", default=False)
		parser.add_argument("--enable-profiling", action="store_true", default=False)


		parser.add_argument("--cluster_forming_threshold", type=float, default=1.00e-07)

		parser.add_argument("--sample_rate", type=float, default=0.01)

		parser.add_argument("--confidence_rate", type=float, default=0.8)

		parser.add_argument("--minibatch_percentage", type=float, default=0.015)

		parser.add_argument("--starting_minibatch", type=int, default=0)

		parser.add_argument("--non_changing_index", type=int, default=0)

		#============================== Drop percentage Target ================================ 
		parser.add_argument("--target_drop_percentage", type=float, default=0.3)

		# =====================================================================================

		args = parser.parse_args()
		print(" ")
		print("#============ Slipstream_Parameters =========#")
		print("Minibatch_Size : ",args.mini_batch_size)
		print("Cluster_Forming_Threshold : ", args.cluster_forming_threshold)
		print("Sample_Rate : ",args.sample_rate)
		print("Confidence_Rate : ", args.confidence_rate)
		print("Minibatch_Percentage : ",args.minibatch_percentage)
		print("Starting_Sampling_Minibatch : ",args.starting_minibatch)
		print("Non_Changing_Indexs : ",args.non_changing_index)
		print("#==============================================#")
		print(" ")

		# the code requires access to dlrm model
		if not path.exists(str(args.dlrm_path)):
				sys.exit("Please provide path to DLRM as --dlrm-path")
		sys.path.insert(1, args.dlrm_path)
		import dlrm_opt_alibaba as dlrm

		if args.datatype == "taobao" and args.arch_embedding_size != "987994-4162024-9439":
				sys.exit(
						"ERROR: arch-embedding-size for taobao "
						+ " needs to be 987994-4162024-9439"
				)
		if args.tsl_inner not in ["def", "ind"] and int(args.tsl_num_heads) > 1:
				 sys.exit(
						"ERROR: dot product "
						+ " assumes one tsl component (due to redundancy)"
				)

		# model_type = "tsl", "mha", "rnn"
		print("dlrm path: ", args.dlrm_path)
		print("model_type: ", args.model_type)
		print("time series length: ", args.ts_length)
		print("seed: ", args.numpy_rand_seed)
		print("model_file:", args.save_model)

		### some basic setup ###
		use_gpu = args.use_gpu and torch.cuda.is_available()
		set_seed(args.numpy_rand_seed, use_gpu)
		np.set_printoptions(precision=args.print_precision)
		torch.set_printoptions(precision=args.print_precision)
		print("use-gpu:", use_gpu)

		# possible modes:
		# "train-test" for both training and metric computation on test data,
		# "train"      for training model
		# "test"       for metric computation on test data using saved trained model
		modes = args.mode.split("-")
		if modes[0] == "train":
				train_tbsm(args, use_gpu)
		if modes[0] == "test" or (len(modes) > 1 and modes[1] == "test"):
				test_tbsm(args, use_gpu)
