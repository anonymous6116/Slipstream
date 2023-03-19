# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: an implementation of a deep learning recommendation model (DLRM)
# The model input consists of dense and sparse features. The former is a vector
# of floating point values. The latter is a list of sparse indices into
# embedding tables, which consist of vectors of floating point values.
# The selected vectors are passed to mlp networks denoted by triangles,
# in some cases the vectors are interacted through operators (Ops).
#
# output:
#                         vector of values
# model:                        |
#                              /\
#                             /__\
#                               |
#       _____________________> Op  <___________________
#     /                         |                      \
#    /\                        /\                      /\
#   /__\                      /__\           ...      /__\
#    |                          |                       |
#    |                         Op                      Op
#    |                    ____/__\_____           ____/__\____
#    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
# input:
# [ dense features ]     [sparse indices] , ..., [sparse indices]
#
# More precise definition of model layers:
# 1) fully connected layers of an mlp
# z = f(y)
# y = Wx + b
#
# 2) embedding lookup (for a list of sparse indices p=[p1,...,pk])
# z = Op(e1,...,ek)
# obtain vectors e1=E[:,p1], ..., ek=E[:,pk]
#
# 3) Operator Op can be one of the following
# Sum(e1,...,ek) = e1 + ... + ek
# Dot(e1,...,ek) = [e1'e1, ..., e1'ek, ..., ek'e1, ..., ek'ek]
# Cat(e1,...,ek) = [e1', ..., ek']'
# where ' denotes transpose operation
#
# References:
# [1] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
# Narayanan Sundaram, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu,
# Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii,
# Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko,
# Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong,
# Misha Smelyanskiy, "Deep Learning Recommendation Model for Personalization and
# Recommendation Systems", CoRR, arXiv:1906.00091, 2019

from __future__ import absolute_import, division, print_function, unicode_literals

# miscellaneous
import builtins
import functools

import time
import json
# data generation
import dlrm_data_pytorch as dp
import dlrm_data_avazu_pytorch as dp_ava

# numpy
import numpy as np
import pandas as pd 

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

import csv

# onnx
# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore", category=DeprecationWarning)


# pytorch
import torch
import torch.nn as nn
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter

# quotient-remainder trick
from tricks.qr_embedding_bag import QREmbeddingBag

# mixed-dimension trick
from tricks.md_embedding_bag import PrEmbeddingBag, md_solver

import sklearn.metrics
from torch.optim.lr_scheduler import _LRScheduler


exc = getattr(builtins, "IOError", "FileNotFoundError")

class LRPolicyScheduler(_LRScheduler):
	def __init__(self, optimizer, num_warmup_steps, decay_start_step, num_decay_steps):
		self.num_warmup_steps = num_warmup_steps
		self.decay_start_step = decay_start_step
		self.decay_end_step = decay_start_step + num_decay_steps
		self.num_decay_steps = num_decay_steps

		if self.decay_start_step < self.num_warmup_steps:
			sys.exit("Learning rate warmup must finish before the decay starts")

		super(LRPolicyScheduler, self).__init__(optimizer)

	def get_lr(self):
		step_count = self._step_count
		if step_count < self.num_warmup_steps:
			# warmup
			scale = 1.0 - (self.num_warmup_steps - step_count) / self.num_warmup_steps
			lr = [base_lr * scale for base_lr in self.base_lrs]
			self.last_lr = lr
		elif self.decay_start_step <= step_count and step_count < self.decay_end_step:
			# decay
			decayed_steps = step_count - self.decay_start_step
			scale = ((self.num_decay_steps - decayed_steps) / self.num_decay_steps) ** 2
			min_lr = 0.0000001
			lr = [max(min_lr, base_lr * scale) for base_lr in self.base_lrs]
			self.last_lr = lr
		else:
			if self.num_decay_steps > 0:
				# freeze at last, either because we're after decay
				# or because we're between warmup and decay
				lr = self.last_lr
			else:
				# do not adjust
				lr = self.base_lrs
		return lr

### define dlrm in PyTorch ###
class DLRM_Net(nn.Module):
	def create_mlp(self, ln, sigmoid_layer):
		# build MLP layer by layer
		layers = nn.ModuleList()
		for i in range(0, ln.size - 1):
			n = ln[i]
			m = ln[i + 1]

			# construct fully connected operator
			LL = nn.Linear(int(n), int(m), bias=True)

			# initialize the weights
			# custom Xavier input, output or two-sided fill
			mean = 0.0  
			std_dev = np.sqrt(2 / (m + n))  
			W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
			std_dev = np.sqrt(1 / m)  
			bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)

			LL.weight.data = torch.tensor(W, requires_grad=True)
			LL.bias.data = torch.tensor(bt, requires_grad=True)
			layers.append(LL)

			# construct sigmoid or relu operator
			if i == sigmoid_layer:
				layers.append(nn.Sigmoid())
			else:
				layers.append(nn.ReLU())


		return torch.nn.Sequential(*layers)

	def create_emb(self, m, ln):
		emb_l = nn.ModuleList()
		for i in range(0, ln.size):
			n = ln[i]
			# construct embedding operator
			if self.qr_flag and n > self.qr_threshold:
				EE = QREmbeddingBag(n, m, self.qr_collisions,
					operation=self.qr_operation, mode="sum", sparse=True)
			elif self.md_flag:
				base = max(m)
				_m = m[i] if n > self.md_threshold else base
				EE = PrEmbeddingBag(n, _m, base)
				# use np initialization as below for consistency...
				W = np.random.uniform(
					low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, _m)
				).astype(np.float32)
				EE.embs.weight.data = torch.tensor(W, requires_grad=True)

			else:
				W = np.random.uniform(
					low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
				).astype(np.float32)

				tensor_EE = torch.tensor(W, requires_grad=True, device="cpu")
				EE = nn.EmbeddingBag(n, m, mode="sum", _weight=tensor_EE, sparse=True)

			emb_l.append(EE)

		return emb_l

	def create_hot_emb(self, m, ln):
		hot_emb_l = nn.ModuleList()

		EE = nn.EmbeddingBag(ln, m, mode="sum", sparse=True)

		# initialize embeddings
		W = np.random.uniform(low=-np.sqrt(1 / ln), high=np.sqrt(1 / ln), size=(ln, m)).astype(np.float32)
		EE.weight.data = torch.tensor(W, requires_grad=True)
		hot_emb_l.append(EE)
		
		return hot_emb_l

	def __init__(
		self,
		m_spa=None,
		ln_emb=None,
		ln_hot_emb = None,
		ln_bot=None,
		ln_top=None,
		arch_interaction_op=None,
		arch_interaction_itself=False,
		sigmoid_bot=-1,
		sigmoid_top=-1,
		sync_dense_params=True,
		loss_threshold=0.0,
		ndevices=-1,
		qr_flag=False,
		qr_operation="mult",
		qr_collisions=0,
		qr_threshold=200,
		md_flag=False,
		md_threshold=200,
	):
		super(DLRM_Net, self).__init__()

		if (
			(m_spa is not None)
			and (ln_emb is not None)
			and (ln_hot_emb is not None)
			and (ln_bot is not None)
			and (ln_top is not None)
			and (arch_interaction_op is not None)
		):

			# save arguments
			self.ndevices = ndevices
			self.output_d = 0
			self.parallel_model_batch_size = -1
			self.parallel_model_is_not_prepared = True
			self.arch_interaction_op = arch_interaction_op
			self.arch_interaction_itself = arch_interaction_itself
			self.sync_dense_params = sync_dense_params
			self.loss_threshold = loss_threshold
			# create variables for QR embedding if applicable
			self.qr_flag = qr_flag
			if self.qr_flag:
				self.qr_collisions = qr_collisions
				self.qr_operation = qr_operation
				self.qr_threshold = qr_threshold
			# create variables for MD embedding if applicable
			self.md_flag = md_flag
			if self.md_flag:
				self.md_threshold = md_threshold
			# create operators
			self.emb_l = self.create_emb(m_spa, ln_emb)
			print("EMB : ", ln_emb)
			self.hot_emb_l = self.create_hot_emb(m_spa, ln_hot_emb)
			print("Hot_EMB : ", ln_hot_emb)
			self.hot_emb_l = self.hot_emb_l.to("cuda:0")
			self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
			self.bot_l = self.bot_l.to("cuda:0")
			self.top_l = self.create_mlp(ln_top, sigmoid_top)
			self.top_l = self.top_l.to("cuda:0")

	def apply_mlp(self, x, layers):
		return layers(x)

	def apply_emb(self, lS_o, lS_i, emb_l):
		# WARNING: notice that we are processing the batch at once. We implicitly
		# assume that the data is laid out such that:
		# 1. each embedding is indexed with a group of sparse indices,
		#   corresponding to a single lookup
		# 2. for each embedding the lookups are further organized into a batch
		# 3. for a list of embedding tables there is a list of batched lookups

		ly = []
		# for k, sparse_index_group_batch in enumerate(lS_i):
		for k in range(len(lS_i)):
			sparse_index_group_batch = lS_i[k]
			sparse_offset_group_batch = lS_o[k]

			# embedding lookup
			# We are using EmbeddingBag, which implicitly uses sum operator.
			# The embeddings are represented as tall matrices, with sum
			# happening vertically across 0 axis, resulting in a row vector
			E = emb_l[k]
			V = E(sparse_index_group_batch, sparse_offset_group_batch)
			ly.append(V)

		return ly

	def apply_hot_emb(self, lS_o, lS_i, emb_l):
		# WARNING: notice that we are processing the batch at once. We implicitly
		# assume that the data is laid out such that:
		# 1. each embedding is indexed with a group of sparse indices,
		#   corresponding to a single lookup
		# 2. for each embedding the lookups are further organized into a batch
		# 3. for a list of embedding tables there is a list of batched lookups
		ly = []
		for k in range(len(lS_i)):
			sparse_index_group_batch = lS_i[k]
			sparse_offset_group_batch = lS_o[k]

			# embedding lookup
			# We are using EmbeddingBag, which implicitly uses sum operator.
			# The embeddings are represented as tall matrices, with sum
			# happening vertically across 0 axis, resulting in a row vector
			E = emb_l[0]
			V = E(sparse_index_group_batch, sparse_offset_group_batch)
			ly.append(V)


		return ly

	def interact_features(self, x, ly):
		if self.arch_interaction_op == "dot":
			# concatenate dense and sparse features
			(batch_size, d) = x.shape
			T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
			# perform a dot product
			Z = torch.bmm(T, torch.transpose(T, 1, 2))
			_, ni, nj = Z.shape
			offset = 1 if self.arch_interaction_itself else 0
			li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
			lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
			Zflat = Z[:, li, lj]
			# concatenate dense features and interactions
			R = torch.cat([x] + [Zflat], dim=1)
		elif self.arch_interaction_op == "cat":
			# concatenation features (into a row vector)
			R = torch.cat([x] + ly, dim=1)
		else:
			sys.exit(
				"ERROR: --arch-interaction-op="
				+ self.arch_interaction_op
				+ " is not supported"
			)

		return R

	def forward(self, dense_x, lS_o, lS_i, data):
				
		if data == "hot":
			return self.parallel_forward(dense_x, lS_o, lS_i)
		else:
			return self.mixed_forward(dense_x, lS_o, lS_i)

	def mixed_forward(self, dense_x, lS_o, lS_i):   
		# Process dense features on GPU in a data parallel fashion
		### prepare model (overwrite) ###
		# WARNING: # of devices must be >= batch size in parallel_forward call
		batch_size = dense_x.size()[0]
		ndevices = min(self.ndevices, batch_size, len(self.emb_l))
		device_ids = range(ndevices)
		# WARNING: must redistribute the model if mini-batch size changes(this is common
		# for last mini-batch, when # of elements in the dataset/batch size is not even
		if self.parallel_model_batch_size != batch_size:
			self.parallel_model_is_not_prepared = True


		if self.parallel_model_is_not_prepared or self.sync_dense_params:
			self.bot_l_replicas = replicate(self.bot_l, device_ids)
			self.top_l_replicas = replicate(self.top_l, device_ids)
			self.parallel_model_batch_size = batch_size


		dense_x = scatter(dense_x, device_ids, dim=0)
		x = parallel_apply(self.bot_l_replicas, dense_x, None, device_ids)

		ly = self.apply_emb(lS_o, lS_i, self.emb_l)
		ly = torch.stack(ly)

		# scattering ly across GPU's
		ly = ly.to("cuda:0")
		t_list = []
		for k, _ in enumerate(self.emb_l):
			y = scatter(ly[k], device_ids, dim=0)
			t_list.append(y)
		# adjust the list to be ordered per device
		ly = list(map(lambda y: list(y), zip(*t_list)))
		
		z = []
		for k in range(ndevices):
			zk = self.interact_features(x[k], ly[k])
			z.append(zk)

		# WARNING: Note that the self.top_l is a list of top mlp modules that
		# have been replicated across devices, while z is a list of interaction results
		# that by construction are scattered across devices on the first (batch) dim.
		# The output is a list of tensors scattered across devices according to the
		# distribution of z.
		p = parallel_apply(self.top_l_replicas, z, None, device_ids)
		### gather the distributed results ###
		p0 = gather(p, self.output_d, dim=0)

		# clamp output if needed
		if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
			z0 = torch.clamp(
				p0, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
			)
		else:
			z0 = p0

		return z0


	def sequential_forward(self, dense_x, lS_o, lS_i):
		# process dense features (using bottom mlp), resulting in a row vector
		x = self.apply_mlp(dense_x, self.bot_l)

		# process sparse features(using embeddings), resulting in a list of row vectors
		ly = self.apply_emb(lS_o, lS_i, self.emb_l)

		# interact features (dense and sparse)
		z = self.interact_features(x, ly)

		# obtain probability of a click (using top mlp)
		p = self.apply_mlp(z, self.top_l)

		# clamp output if needed
		if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
			z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
		else:
			z = p

		return z

	def parallel_forward(self, dense_x, lS_o, lS_i):

		batch_size = dense_x.size()[0]
		ndevices = min(self.ndevices, batch_size)
		device_ids = range(ndevices)
		# WARNING: must redistribute the model if mini-batch size changes(this is common
		# for last mini-batch, when # of elements in the dataset/batch size is not even
		if self.parallel_model_batch_size != batch_size:
			self.parallel_model_is_not_prepared = True

		if self.parallel_model_is_not_prepared or self.sync_dense_params:
			self.bot_l_replicas = replicate(self.bot_l, device_ids)
			self.top_l_replicas = replicate(self.top_l, device_ids)
			self.hot_emb_l_replicas = replicate(self.hot_emb_l, device_ids)
			self.parallel_model_batch_size = batch_size

		dense_x = scatter(dense_x, device_ids, dim=0)

		
		lS_i = scatter(lS_i, device_ids, dim=1)
		lS_o = scatter(lS_o, device_ids, dim=1)
		lS_o_t = lS_o[0]
		lS_o = []
		for i in range(ndevices):
			lS_o.append(lS_o_t.to("cuda:"+str(i)))

		### compute results in parallel ###
		# bottom mlp
		# WARNING: Note that the self.bot_l is a list of bottom mlp modules
		# that have been replicated across devices, while dense_x is a tuple of dense
		# inputs that has been scattered across devices on the first (batch) dimension.
		# The output is a list of tensors scattered across devices according to the
		# distribution of dense_x.
		x = parallel_apply(self.bot_l_replicas, dense_x, None, device_ids)
		ly = []
		
		for i in range(ndevices):
			y = self.apply_hot_emb(lS_o[i], lS_i[i], self.hot_emb_l_replicas[i])
			ly.append(y)

		
		# butterfly shuffle (implemented inefficiently for now)
		# WARNING: Note that at this point we have the result of the embedding lookup
		# for the entire batch on each device. We would like to obtain partial results
		# corresponding to all embedding lookups, but part of the batch on each device.
		# Therefore, matching the distribution of output of bottom mlp, so that both
		# could be used for subsequent interactions on each device.

		# interactions
		z = []
		for k in range(ndevices):
			zk = self.interact_features(x[k], ly[k])
			z.append(zk)
		
		# top mlp
		# WARNING: Note that the self.top_l is a list of top mlp modules that
		# have been replicated across devices, while z is a list of interaction results
		# that by construction are scattered across devices on the first (batch) dim.
		# The output is a list of tensors scattered across devices according to the
		# distribution of z.

		p = parallel_apply(self.top_l_replicas, z, None, device_ids)

		### gather the distributed results ###
		p0 = gather(p, self.output_d, dim=0)
			
		# clamp output if needed
		if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
			z0 = torch.clamp(
				p0, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
			)
		else:
			z0 = p0

		return z0

def dash_separated_ints(value):
	vals = value.split('-')
	for val in vals:
		try:
			int(val)
		except ValueError:
			raise argparse.ArgumentTypeError(
				"%s is not a valid dash separated list of ints" % value)

	return value


def dash_separated_floats(value):
	vals = value.split('-')
	for val in vals:
		try:
			float(val)
		except ValueError:
			raise argparse.ArgumentTypeError(
				"%s is not a valid dash separated list of floats" % value)

	return value


if __name__ == "__main__":
	### import packages ###
	import sys
	import argparse

	### parse arguments ###
	parser = argparse.ArgumentParser(
		description="Train Deep Learning Recommendation Model (DLRM)"
	)
	# model related parameters
	parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
	parser.add_argument(
		"--arch-embedding-size", type=dash_separated_ints, default="4-3-2")
	# j will be replaced with the table number
	parser.add_argument(
		"--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
	parser.add_argument(
		"--arch-mlp-top", type=dash_separated_ints, default="4-2-1")
	parser.add_argument(
		"--arch-interaction-op", type=str, choices=['dot', 'cat'], default="dot")
	parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
	# embedding table options
	parser.add_argument("--md-flag", action="store_true", default=False)
	parser.add_argument("--md-threshold", type=int, default=200)
	parser.add_argument("--md-temperature", type=float, default=0.3)
	parser.add_argument("--md-round-dims", action="store_true", default=False)
	parser.add_argument("--qr-flag", action="store_true", default=False)
	parser.add_argument("--qr-threshold", type=int, default=200)
	parser.add_argument("--qr-operation", type=str, default="mult")
	parser.add_argument("--qr-collisions", type=int, default=4)
	# activations and loss
	parser.add_argument("--activation-function", type=str, default="relu")
	parser.add_argument("--loss-function", type=str, default="mse")  # or bce or wbce
	parser.add_argument(
		"--loss-weights", type=dash_separated_floats, default="1.0-1.0")  # for wbce
	parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
	parser.add_argument("--round-targets", type=bool, default=False)
	# data
	parser.add_argument("--data-size", type=int, default=1)
	parser.add_argument("--num-batches", type=int, default=0)
	parser.add_argument(
		"--data-generation", type=str, default="random"
	)  # synthetic or dataset
	parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
	parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
	parser.add_argument("--raw-data-file", type=str, default="")
	parser.add_argument("--processed-data-file", type=str, default="")
	# ========================= Added Avazu Train and Test Files ========================
	parser.add_argument("--avazu-db-path", type=str, default="")
	parser.add_argument("--avazu-train-file", type=str, default="") # avazu_train.npz
	parser.add_argument("--avazu-test-file", type=str, default="") # avazu_test.npz
	# ===================================================================================
	# ========================= Added train files and dict ==============================
	parser.add_argument("--train-hot-file", type=str, default="") # train_hot.npz
	parser.add_argument("--train-normal-file", type=str, default="") # train_normal.npz
	parser.add_argument("--hot-emb-dict-file", type=str, default="") # hot_emb_dict.npz
	# ===================================================================================
	parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
	parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
	parser.add_argument("--max-ind-range", type=int, default=-1)
	parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
	parser.add_argument("--num-indices-per-lookup", type=int, default=10)
	parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--memory-map", action="store_true", default=False)
	parser.add_argument("--dataset-multiprocessing", action="store_true", default=False,
						help="The Kaggle dataset can be multiprocessed in an environment \
						with more than 7 CPU cores and more than 20 GB of memory. \n \
						The Terabyte dataset can be multiprocessed in an environment \
						with more than 24 CPU cores and at least 1 TB of memory.")
	# training
	parser.add_argument("--mini-batch-size", type=int, default=1)
	parser.add_argument("--nepochs", type=int, default=1)
	parser.add_argument("--learning-rate", type=float, default=0.01)
	parser.add_argument("--print-precision", type=int, default=5)
	parser.add_argument("--numpy-rand-seed", type=int, default=123)
	parser.add_argument("--sync-dense-params", type=bool, default=True)
	# inference
	parser.add_argument("--inference-only", action="store_true", default=False)
	# onnx
	parser.add_argument("--save-onnx", action="store_true", default=False)
	# gpu
	parser.add_argument("--use-gpu", action="store_true", default=True)
	# debugging and profiling
	parser.add_argument("--print-freq", type=int, default=1)
	parser.add_argument("--test-freq", type=int, default=-1)
	parser.add_argument("--test-mini-batch-size", type=int, default=-1)
	parser.add_argument("--test-num-workers", type=int, default=-1)
	parser.add_argument("--print-time", action="store_true", default=False)
	parser.add_argument("--debug-mode", action="store_true", default=False)
	parser.add_argument("--enable-profiling", action="store_true", default=False)
	parser.add_argument("--plot-compute-graph", action="store_true", default=False)
	# store/load model
	parser.add_argument("--save-model", type=str, default="")
	parser.add_argument("--load-model", type=str, default="")
	# mlperf logging (disables other output and stops early)
	parser.add_argument("--mlperf-logging", action="store_true", default=False)
	# stop at target accuracy Kaggle 0.789, Terabyte (sub-sampled=0.875) 0.8107
	# default has been changed by Yass for both below 
	parser.add_argument("--mlperf-acc-threshold", type=float, default=90.0)
	# stop at target AUC Terabyte (no subsampling) 0.8025
	parser.add_argument("--mlperf-auc-threshold", type=float, default=90.0)
	parser.add_argument("--mlperf-bin-loader", action='store_true', default=False)
	parser.add_argument("--mlperf-bin-shuffle", action='store_true', default=False)
	# LR policy
	parser.add_argument("--lr-num-warmup-steps", type=int, default=0)
	parser.add_argument("--lr-decay-start-step", type=int, default=0)
	parser.add_argument("--lr-num-decay-steps", type=int, default=0)

	# ======================= Slipstream parameters ====================================

	# threshold to segregate changing and non-changing hot entries 
	parser.add_argument("--cluster_forming_threshold", type=float, default=1.00e-07)

	# percentage hot embedding table sampling 
	parser.add_argument("--sample_rate", type=float, default=0.01)

	# threshold above which we stop the sampling process and move on with regular hot training 
	parser.add_argument("--confidence_rate", type=float, default=0.8)

	# percentage of overall number of minibatches used for sampling phase
	parser.add_argument("--minibatch_percentage", type=float, default=0.015) 

	# minibatch t0 start sampling 
	parser.add_argument("--starting_minibatch", type=int, default=0) 

	# number of non_changing indices to have for an input to be considered non-changing input -- kaggle & terabyte 26
	parser.add_argument("--non_changing_index", type=int, default=26) 


	#============================== Drop percentage Target ================================ 
	parser.add_argument("--target_drop_percentage", type=float, default=0.3)

	# =====================================================================================
	args = parser.parse_args()

	print(" ")
	print("# ============ Slipstream_Parameters ============= #")
	print(" Minibatch_Size : ",args.mini_batch_size)
	print(" Cluster_Forming_Threshold : ", args.cluster_forming_threshold)
	print(" Sample_Rate : ",args.sample_rate)
	print(" Confidence_Rate : ", args.confidence_rate)
	print(" Minibatch_Percentage : ",args.minibatch_percentage)
	print(" Starting_Sampling_Minibatch : ",args.starting_minibatch)
	print(" Non_Changing_Indexs : ",args.non_changing_index)
	print(" Target_Drop_Percentage : ",args.target_drop_percentage)
	print("# ================================================ #")
	print(" ")
			
	if args.mlperf_logging:
		print('command_line_args : ', json.dumps(vars(args)))

	### some basic setup ###
	np.random.seed(args.numpy_rand_seed)
	np.set_printoptions(precision=args.print_precision)
	torch.set_printoptions(precision=args.print_precision)
	torch.manual_seed(args.numpy_rand_seed)

	if (args.test_mini_batch_size < 0):
		# if the parameter is not set, use the training batch size
		args.test_mini_batch_size = args.mini_batch_size
	if (args.test_num_workers < 0):
		# if the parameter is not set, use the same parameter for training
		args.test_num_workers = args.num_workers

	use_gpu = args.use_gpu and torch.cuda.is_available()
	### main loop ###
	def time_wrap(use_gpu):
		if use_gpu:
			torch.cuda.synchronize()
		return time.time()

	if use_gpu:
		torch.cuda.manual_seed_all(args.numpy_rand_seed)
		torch.backends.cudnn.deterministic = True
		device = torch.device("cuda", 0)
		ngpus = torch.cuda.device_count()  

		time_running_1 = time_wrap(use_gpu)
		print("DLRM_Slipstream_GPU")
		print("Start_Operations_Time_Sec : 0")
		print("CPU_{}GPUs".format(ngpus))
	else:
		time_running_1 = time_wrap(use_gpu)
		device = torch.device("cpu")
		print("DLRM_Slipstream_CPU")
		print("Start_Operations_Time_Sec : 0")
		print("CPU")

	print("Dataset : ", args.data_set)
	print("Threshold to start for the first epoch : ",args.cluster_forming_threshold)
	ending_minibatch = 0

	### prepare training data ###
	ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
	# input data
	if (args.data_generation == "dataset"):
		if (args.data_set == "kaggle" or args.data_set == "terabyte"):
			# =================== Commenting actual dataset load and using just test data =====================
			train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)

			# =================================================================================================

			# ============================== Loading processed hot data and normal data =======================
			print("Loading_preprocessed_Data")


			train_normal = np.load(args.train_normal_file, allow_pickle = True)
			train_normal = train_normal['arr_0']
			train_normal = train_normal.tolist()
			print("Elements_Train_Cold : ", len(train_normal))

			train_hot = np.load(args.train_hot_file, allow_pickle = True)
			train_hot = train_hot['arr_0']
			train_hot = train_hot.tolist()
			print("Elements_Train_Hot : ", len(train_hot))

			hot_emb_dict = np.load(args.hot_emb_dict_file, allow_pickle = True)
			hot_emb_dict = hot_emb_dict['arr_0']
			hot_emb_dict = hot_emb_dict.tolist()
			print("Length_Hot_Dictionary : ", len(hot_emb_dict))

			ln_hot_emb = 0
			for i, dict in enumerate(hot_emb_dict):
				ln_hot_emb  = ln_hot_emb + len(dict)
			print("Elements_Hot_Dictionary : ", ln_hot_emb)
			time_running_2 = time_wrap(use_gpu) 
			print("File_Operations_Time_Sec : ", time_running_2 - time_running_1)   
			

			train_hot_ld, train_normal_ld = dp.load_criteo_preprocessed_data_and_loaders(args, train_hot, train_normal)
		
			# ===================================================================================================
			nbatches_hot = args.num_batches if args.num_batches > 0 else len(train_hot_ld)
			nbatches_normal = args.num_batches if args.num_batches > 0 else len(train_normal_ld)
			nbatches = nbatches_hot + nbatches_normal
			
			ending_minibatch = round(args.minibatch_percentage * nbatches_hot) + args.starting_minibatch
			nbatches_test = len(test_ld)


			print(" ")
			print("#__________________________________________#")
			print("Minibatch_Size : ", args.mini_batch_size)
			print("Number_Hot_Minibatches : ", nbatches_hot)
			print("Number_Cold_Minibatches : ", nbatches_normal)
			print("Total_Minibatches : ", nbatches)
			print("Test_Minibatches : ", nbatches_test)
			print("Ending_Minibatch : ",ending_minibatch)
			print("Starting_Minibatch : ",args.starting_minibatch)
			print("#__________________________________________#")
			print(" ")


			ln_emb = train_data.counts

			# enforce maximum limit on number of vectors per embedding
			if args.max_ind_range > 0:
				ln_emb = np.array(list(map(
					lambda x: x if x < args.max_ind_range else args.max_ind_range,
					ln_emb
				)))

			m_den = len(train_normal[0][0])
			ln_bot[0] = m_den
		
		elif (args.data_set == "avazu"):

			train_data, train_ld, test_data, test_ld = dp_ava.make_avazu_data_and_loaders(args)

			# ============================== Loading processed hot data and normal data =======================
			print("Loading_preprocessed_Data_Avazu")

			
			train_normal = np.load(args.train_normal_file, allow_pickle = True)
			train_normal = train_normal['arr_0']
			train_normal = train_normal.tolist()
			print("Elements_Train_Cold : ", len(train_normal))

			train_hot = np.load(args.train_hot_file, allow_pickle = True)
			train_hot = train_hot['arr_0']
			train_hot = train_hot.tolist()
			print("Elements_Train_Hot : ", len(train_hot))
			

			hot_emb_dict = np.load(args.hot_emb_dict_file, allow_pickle = True)
			hot_emb_dict = hot_emb_dict['arr_0']
			hot_emb_dict = hot_emb_dict.tolist()
			print("Length_Hot_Dictionary : ", len(hot_emb_dict))

			ln_hot_emb = 0
			for i, dict in enumerate(hot_emb_dict):
				ln_hot_emb  = ln_hot_emb + len(dict)

			print("Elements_Hot_Dictionary : ", ln_hot_emb)
			time_running_2 = time_wrap(use_gpu) 
			print("File_Operations_Time_Sec : ", time_running_2 - time_running_1)
			
			train_hot_ld, train_normal_ld = dp_ava.load_avazu_preprocessed_data_and_loaders(args, train_hot, train_normal)

			# ===================================================================================================

			nbatches_hot = args.num_batches if args.num_batches > 0 else len(train_hot_ld)
			nbatches_normal = args.num_batches if args.num_batches > 0 else len(train_normal_ld)
			nbatches = nbatches_hot + nbatches_normal

		   
			ending_minibatch = round(args.minibatch_percentage * nbatches_hot) + args.starting_minibatch
			nbatches_test = len(test_ld)

			print(" ")
			print("#_________________________________________#")
			print("Minibatch_Size : ", args.mini_batch_size)
			print("Number_Hot_Minibatches : ", nbatches_hot)
			print("Number_Cold_Minibatches : ", nbatches_normal)
			print("Ending_Minibatch : ",ending_minibatch)
			print("Total_Minibatches : ", nbatches)
			print("Test_Minibatches : ", nbatches_test)
			print("#_________________________________________#")
			print(" ")

			ln_emb = train_data.counts
			
			# enforce maximum limit on number of vectors per embedding
			if args.max_ind_range > 0:
				ln_emb = np.array(list(map(
					lambda x: x if x < args.max_ind_range else args.max_ind_range,
					ln_emb
				)))
			m_den = train_data.m_den
			ln_bot[0] = m_den
			
			
			print("Loading_Avazu_Test_Data")
			test_data = np.load(args.avazu_test_file, allow_pickle = True)
			test_data = test_data['arr_0']
			test_data = test_data.tolist()
			print("\nLength_Avazu_Test_Data : ", len(test_data))
			
			test_ld = dp_ava.load_avazu_test_data_and_loaders(args, test_data)


	else:
		# input and target at random
		ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
		m_den = ln_bot[0]
		train_data, train_ld = dp.make_random_data_and_loader(args, ln_emb, m_den)
		nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)

	### parse command line arguments ###
	m_spa = args.arch_sparse_feature_size
	num_fea = ln_emb.size + 1  # num sparse + num dense features
	m_den_out = ln_bot[ln_bot.size - 1]
	if args.arch_interaction_op == "dot":
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

	# sanity check: feature sizes and mlp dimensions must match
	if m_den != ln_bot[0]:
		sys.exit(
			"ERROR: arch-dense-feature-size "
			+ str(m_den)
			+ " does not match first dim of bottom mlp "
			+ str(ln_bot[0])
		)
	if args.qr_flag:
		if args.qr_operation == "concat" and 2 * m_spa != m_den_out:
			sys.exit(
				"ERROR: 2 arch-sparse-feature-size "
				+ str(2 * m_spa)
				+ " does not match last dim of bottom mlp "
				+ str(m_den_out)
				+ " (note that the last dim of bottom mlp must be 2x the embedding dim)"
			)
		if args.qr_operation != "concat" and m_spa != m_den_out:
			sys.exit(
				"ERROR: arch-sparse-feature-size "
				+ str(m_spa)
				+ " does not match last dim of bottom mlp "
				+ str(m_den_out)
			)
	else:
		if m_spa != m_den_out:
			sys.exit(
				"ERROR: arch-sparse-feature-size "
				+ str(m_spa)
				+ " does not match last dim of bottom mlp "
				+ str(m_den_out)
			)
	if num_int != ln_top[0]:
		sys.exit(
			"ERROR: # of feature interactions "
			+ str(num_int)
			+ " does not match first dimension of top mlp "
			+ str(ln_top[0])
		)

	# assign mixed dimensions if applicable
	if args.md_flag:
		m_spa = md_solver(
			torch.tensor(ln_emb),
			args.md_temperature,  # alpha
			d0=m_spa,
			round_dim=args.md_round_dims
		).tolist()

	if args.debug_mode:
		print("model arch:")
		print(
			"mlp top arch "
			+ str(ln_top.size - 1)
			+ " layers, with input to output dimensions:"
		)
		print(ln_top)
		print("# of interactions")
		print(num_int)
		print(
			"mlp bot arch "
			+ str(ln_bot.size - 1)
			+ " layers, with input to output dimensions:"
		)
		print(ln_bot)
		print("# of features (sparse and dense)")
		print(num_fea)
		print("dense feature size")
		print(m_den)
		print("sparse feature size")
		print(m_spa)
		print(
			"# of embeddings (= # of sparse features) "
			+ str(ln_emb.size)
			+ ", with dimensions "
			+ str(m_spa)
			+ "x:"
		)
		print(ln_emb)

		print("Data_inputs_and_targets : ")
		for j, (X, lS_o, lS_i, T) in enumerate(train_ld):
			# early exit if nbatches was set by the user and has been exceeded
			if nbatches > 0 and j >= nbatches:
				break

			print("mini-batch: %d" % j)
			print(X.detach().cpu().numpy())
			# transform offsets to lengths when printing
			print(
				[
					np.diff(
						S_o.detach().cpu().tolist() + list(lS_i[i].shape)
					).tolist()
					for i, S_o in enumerate(lS_o)
				]
			)
			print([S_i.detach().cpu().tolist() for S_i in lS_i])
			print(T.detach().cpu().numpy())

	ndevices = min(ngpus, args.mini_batch_size, num_fea - 1) if use_gpu else -1
	print(" number of GPUs",ndevices)
	### construct the neural network specified above ###
	# WARNING: to obtain exactly the same initialization for
	# the weights we need to start from the same random seed.
	# np.random.seed(args.numpy_rand_seed)
	dlrm = DLRM_Net(
		m_spa,
		ln_emb,
		ln_hot_emb,
		ln_bot,
		ln_top,
		arch_interaction_op=args.arch_interaction_op,
		arch_interaction_itself=args.arch_interaction_itself,
		sigmoid_bot=-1,
		sigmoid_top=ln_top.size - 2,
		sync_dense_params=args.sync_dense_params,
		loss_threshold=args.loss_threshold,
		ndevices=ndevices,
		qr_flag=args.qr_flag,
		qr_operation=args.qr_operation,
		qr_collisions=args.qr_collisions,
		qr_threshold=args.qr_threshold,
		md_flag=args.md_flag,
		md_threshold=args.md_threshold,
	)
	# test prints
	if args.debug_mode:
		print("Initial_parameters_weights_bias : ")
		for param in dlrm.parameters():
			print(param.detach().cpu().numpy())

	#if use_gpu:
		# Custom Model-Data Parallel
		# the mlps are replicated and use data parallelism, while
		# the embeddings are distributed and use model parallelism

	# specify the loss function
	if args.loss_function == "mse":
		loss_fn = torch.nn.MSELoss(reduction="mean")
	elif args.loss_function == "bce":
		loss_fn = torch.nn.BCELoss(reduction="mean")
	elif args.loss_function == "wbce":
		loss_ws = torch.tensor(np.fromstring(args.loss_weights, dtype=float, sep="-"))
		loss_fn = torch.nn.BCELoss(reduction="none")
	else:
		sys.exit("ERROR: --loss-function=" + args.loss_function + " is not supported")

	if not args.inference_only:
		# specify the optimizer algorithm
		optimizer = torch.optim.SGD(dlrm.parameters(), lr=args.learning_rate)
		lr_scheduler = LRPolicyScheduler(optimizer, args.lr_num_warmup_steps, args.lr_decay_start_step,
										 args.lr_num_decay_steps)



	def dlrm_wrap(X, lS_o, lS_i, use_gpu, device, data):
		if data == "hot":  # .cuda()
			# lS_i can be either a list of tensors or a stacked tensor.
			# Handle each case below:
			lS_i = [S_i.to(device) for S_i in lS_i] if isinstance(lS_i, list) \
				else lS_i.to(device)
			lS_o = [S_o.to(device) for S_o in lS_o] if isinstance(lS_o, list) \
				else lS_o.to(device)
			return dlrm(
				X.to(device),
				lS_o,
				lS_i,
				data
			)
		else:
			return dlrm(
				X.to(device),
				lS_o,
				lS_i,
				data
			)

	def loss_fn_wrap(Z, T, use_gpu, device):
		if args.loss_function == "mse" or args.loss_function == "bce":
			if use_gpu:
				return loss_fn(Z, T.to(device))
			else:
				return loss_fn(Z, T)
		elif args.loss_function == "wbce":
			if use_gpu:
				loss_ws_ = loss_ws[T.data.view(-1).long()].view_as(T).to(device)
				loss_fn_ = loss_fn(Z, T.to(device))
			else:
				loss_ws_ = loss_ws[T.data.view(-1).long()].view_as(T)
				loss_fn_ = loss_fn(Z, T.to(device))
			loss_sc_ = loss_ws_ * loss_fn_
			return loss_sc_.mean()

	# training or inference
	best_gA_test = 0
	best_auc_test = 0
	skip_upto_epoch = 0
	skip_upto_batch = 0

	total_time = 0
	total_loss = 0
	total_accu = 0
	total_iter = 0
	total_samp = 0

	forward_time = 0
	backward_time = 0
	optimizer_time = 0
	scheduler_time = 0

	cold_forward_time = 0
	cold_backward_time = 0
	cold_optimizer_time = 0
	cold_scheduler_time = 0

	hot_forward_time = 0
	hot_backward_time = 0
	hot_optimizer_time = 0
	hot_scheduler_time = 0

	fwd_itr = 0
	bwd_itr = 0
	opt_itr = 0
	sch_itr = 0

	cold_total = 0
	hot_total = 0
	full_total = 0

	hot_emb_update = 0
	cold_emb_update = 0

	# Load model is specified
	if not (args.load_model == ""):
		print("Loading_saved_model_{}".format(args.load_model))
		if use_gpu:
			if dlrm.ndevices > 1:
				# NOTE: when targeting inference on multiple GPUs,
				# load the model as is on CPU or GPU, with the move
				# to multiple GPUs to be done in parallel_forward
				ld_model = torch.load(args.load_model)
			else:
				# NOTE: when targeting inference on single GPU,
				# note that the call to .to(device) has already happened
				ld_model = torch.load(
					args.load_model,
					map_location=torch.device('cuda')
				)
		else:
			# when targeting inference on CPU
			ld_model = torch.load(args.load_model, map_location=torch.device('cpu'))
		dlrm.load_state_dict(ld_model["state_dict"])
		ld_j = ld_model["iter"]
		ld_k = ld_model["epoch"]
		ld_nepochs = ld_model["nepochs"]
		ld_nbatches = ld_model["nbatches"]
		ld_nbatches_test = ld_model["nbatches_test"]
		ld_gA = ld_model["train_acc"]
		ld_gL = ld_model["train_loss"]
		ld_total_loss = ld_model["total_loss"]
		ld_total_accu = ld_model["total_accu"]
		ld_gA_test = ld_model["test_acc"]
		ld_gL_test = ld_model["test_loss"]
		if not args.inference_only:
			optimizer.load_state_dict(ld_model["opt_state_dict"])
			best_gA_test = ld_gA_test
			total_loss = ld_total_loss
			total_accu = ld_total_accu
			skip_upto_epoch = ld_k  # epochs
			skip_upto_batch = ld_j  # batches
		else:
			args.print_freq = ld_nbatches
			args.test_freq = 0

		print(
			"Saved at: epoch = {:d}/{:d}, batch = {:d}/{:d}, ntbatch = {:d}".format(
				ld_k, ld_nepochs, ld_j, ld_nbatches, ld_nbatches_test
			)
		)
		print(
			"Training state: loss = {:.6f}, accuracy = {:3.3f} %".format(
				ld_gL, ld_gA * 100
			)
		)
		print(
			"Testing state: loss = {:.6f}, accuracy = {:3.3f} %".format(
				ld_gL_test, ld_gA_test * 100
			)
		)


	best_gA_test = 0
	best_auc_test = 0
	skip_upto_epoch = 0
	skip_upto_batch = 0

	total_time = 0
	total_loss = 0
	total_accu = 0
	total_iter = 0
	total_samp = 0

	forward_time = 0
	backward_time = 0
	optimizer_time = 0
	scheduler_time = 0

	forward_normal_time = 0
	backward_normal_time = 0
	optimizer_normal_time = 0
	scheduler_normal_time = 0

	forward_sampling_time = 0
	backward_sampling_time = 0
	optimizer_sampling_time = 0
	scheduler_sampling_time = 0

	forward_hot_time = 0
	backward_hot_time = 0
	optimizer_hot_time = 0
	scheduler_hot_time = 0

	# slipstream parameters
	cpu_operation_time = 0
	gpu_snapshot_time = 0
	gpu_operation_time = 0

	fwd_itr = 0
	bwd_itr = 0
	opt_itr = 0
	sch_itr = 0

	cold_total = 0
	hot_total = 0
	full_total = 0

	test_time_cold = 0
	test_time_hot = 0

	k = 0
	stop = 0

	# shared memory 
	shm_0 = shared_memory.SharedMemory(create=True, size=40 * 4 * len(train_hot))
	cluster_0 = np.ndarray(len(train_hot), dtype=object, buffer=shm_0.buf)

	shm_1 = shared_memory.SharedMemory(create=True, size=40 * 4 * len(train_hot))
	cluster_1 = np.ndarray(len(train_hot), dtype=object, buffer=shm_1.buf)

	def worker_func(train_hot,non_changing_index,final_total, cluster_0, cluster_1, chunksize):
		clstr_0 = 0
		clstr_1 = 0
		index = int(current_process().name)

		for i, hot_tuple in enumerate(train_hot):
			lS_i_temp = []
			for j, lS_i in enumerate(hot_tuple[1]):
				if final_total[int(lS_i)] == 1:
					lS_i_temp.append(int(lS_i))


			if (len(lS_i_temp) >= non_changing_index):
				# skipable inputs as there are not changing above threshold
				cluster_1[index*chunksize + clstr_1] = hot_tuple         
				clstr_1 +=  1
			else:
				# non-skippable inputs
				cluster_0[index*chunksize + clstr_0] = hot_tuple
				clstr_0 +=  1


	time_running_1 = time_wrap(use_gpu)
	print("Start_Train_Time_Sec : ", time_running_1 - time_running_2)
	print("Start_Train")

	hot_entry_samples = int(args.sample_rate * ln_hot_emb)

	while k < args.nepochs:
		if k < skip_upto_epoch:
			continue

		accum_time_begin = time_wrap(use_gpu)
		cluster_forming_threshold = args.cluster_forming_threshold

		if stop == 1:
			break
		
		
		# Using Normal Train Data           
		for j, (X, lS_o, lS_i, T) in enumerate(train_normal_ld):
			data = "normal"

			if j < skip_upto_batch:
				continue

			t1 = time_wrap(use_gpu)
			
			# early exit if nbatches was set by the user and has been exceeded
			if nbatches_normal > 0 and j >= nbatches_normal:
				break
			
			should_print = ((j + 1) % args.print_freq == 0) or (j + 1 == nbatches_normal)
			
			begin_forward = time_wrap(use_gpu)
			# forward pass
			
			Z = dlrm_wrap(X, lS_o, lS_i, use_gpu, device, data)

			end_forward = time_wrap(use_gpu)

			# loss
			E = loss_fn_wrap(Z, T, use_gpu, device)
			
			# compute loss and accuracy
			L = E.detach().cpu().numpy()  # numpy array
			S = Z.detach().cpu().numpy()  # numpy array
			T = T.detach().cpu().numpy()  # numpy array
			mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
			A = np.sum((np.round(S, 0) == T).astype(np.uint8))

			if not args.inference_only:
				# scaled error gradient propagation
				# (where we do not accumulate gradients across mini-batches)
				optimizer.zero_grad()
				# backward pass
				E.backward()
				end_backward = time_wrap(use_gpu)

				# optimizer
				optimizer.step()

				end_optimizing = time_wrap(use_gpu)

				lr_scheduler.step()

				end_scheduling = time_wrap(use_gpu)

			
			t2 = time_wrap(use_gpu)

			total_accu += A
			total_loss += L * mbs
			total_iter += 1
			total_samp += mbs

			fwd_itr += end_forward - begin_forward
			bwd_itr += end_backward - end_forward
			opt_itr += end_optimizing - end_backward
			sch_itr += end_scheduling - end_optimizing

			cold_forward_time += end_forward - begin_forward
			cold_backward_time += end_backward - end_forward
			cold_optimizer_time += end_optimizing - end_backward
			cold_scheduler_time += end_scheduling - end_optimizing

			cold_total += cold_forward_time
			cold_total += cold_backward_time
			cold_total += cold_optimizer_time
			cold_total += cold_scheduler_time

			should_print = ((j + 1) % args.print_freq == 0) or (j + 1 == nbatches)
			should_test = (
				(args.test_freq > 0)
				and (args.data_generation == "dataset")
				and (((j + 1) % args.test_freq == 0) or (j + 1 == nbatches_normal))
			)

			# print time, loss and accuracy
			if should_print or should_test:

				gT = 1000.0 * total_time / total_iter if args.print_time else -1
				total_time = 0

				gA = total_accu / total_samp
				total_accu = 0

				gL = total_loss / total_samp
				total_loss = 0

				gForward = 1000 * fwd_itr / total_iter

				gBackward = 1000 * bwd_itr / total_iter

				gOptimizer = 1000 * opt_itr / total_iter

				gScheduler = 1000 * sch_itr /total_iter

				str_run_type = "inference" if args.inference_only else "training"

				#print("Cold_Train_Data_" + (j+1) + " : ", data)
				print("Cold_Train_Epoch_"+str(j+1)," : ", k)
				print("Cold_Iteration : " + str(j+1))
				print("Cold_Loss_" + str(j+1) + " : " + str(gL))
				print("Cold_Train_Accuracy_" + str(j+1) + " : " + str(gA*100))
				print("\n")

				
				# Uncomment the line below to print out the total time with overhead
				total_iter = 0
				total_samp = 0

				fwd_itr = 0
				bwd_itr = 0
				opt_itr = 0
				sch_itr = 0                 

			
			# testing
			if should_test and not args.inference_only:
				# don't measure training iter time in a test iteration

				test_accu = 0
				test_loss = 0
				test_samp = 0

				if args.mlperf_logging:
					scores = []
					targets = []

				accum_test_time_begin = time_wrap(use_gpu)
				
				for i, (X_test, lS_o_test, lS_i_test, T_test) in enumerate(test_ld):
					data = "test"
					# early exit if nbatches was set by the user and was exceeded
					if nbatches > 0 and i >= nbatches:
						break

					t1_test = time_wrap(use_gpu)
					should_print = 0
					# forward pass
					Z_test = dlrm_wrap(X_test, lS_o_test, lS_i_test, use_gpu, device, data)

					if args.mlperf_logging:
						S_test = Z_test.detach().cpu().numpy()  # numpy array
						T_test = T_test.detach().cpu().numpy()  # numpy array

						scores.append(S_test)
						targets.append(T_test)


					else:
						# loss
						E_test = loss_fn_wrap(Z_test, T_test, use_gpu, device)

						# compute loss and accuracy
						L_test = E_test.detach().cpu().numpy()  # numpy array
						S_test = Z_test.detach().cpu().numpy()  # numpy array
						T_test = T_test.detach().cpu().numpy()  # numpy array
						mbs_test = T_test.shape[0]  # = mini_batch_size except last
						A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))
						test_accu += A_test
						test_loss += L_test * mbs_test
						test_samp += mbs_test
						


					t2_test = time_wrap(use_gpu)
					test_time_cold += (t2_test - t1_test)   


				if args.mlperf_logging:

					scores = np.concatenate(scores, axis=0)
					targets = np.concatenate(targets, axis=0)
					
					metrics = {
						'loss' : sklearn.metrics.log_loss,
						'recall' : lambda y_true, y_score:
						sklearn.metrics.recall_score(
							y_true=y_true,
							y_pred=np.round(y_score)
						),
						'precision' : lambda y_true, y_score:
						sklearn.metrics.precision_score(
							y_true=y_true,
							y_pred=np.round(y_score)
						),
						'f1' : lambda y_true, y_score:
						sklearn.metrics.f1_score(
							y_true=y_true,
							y_pred=np.round(y_score)
						),
						'ap' : sklearn.metrics.average_precision_score,
						'roc_auc' : sklearn.metrics.roc_auc_score,
						'accuracy' : lambda y_true, y_score:
						sklearn.metrics.accuracy_score(
							y_true=y_true,
							y_pred=np.round(y_score)
						),
						# 'pre_curve' : sklearn.metrics.precision_recall_curve,
						# 'roc_curve' :  sklearn.metrics.roc_curve,
					}

					# print("Compute time for validation metric : ", end="")
					# first_it = True
					validation_results = {}
					for metric_name, metric_function in metrics.items():
						# if first_it:
						#     first_it = False
						# else:
						#     print(", ", end="")
						# metric_compute_start = time_wrap(False)
						validation_results[metric_name] = metric_function(
							targets,
							scores
						)
						# metric_compute_end = time_wrap(False)
						# met_time = metric_compute_end - metric_compute_start
						# print("{} {:.4f}".format(metric_name, 1000 * (met_time)),
						#      end="")
					# print(" ms")
					gA_test = validation_results['accuracy']
					gL_test = validation_results['loss']
				else:
					gA_test = test_accu / test_samp
					gL_test = test_loss / test_samp

				is_best = gA_test > best_gA_test
				if is_best:
					best_gA_test = gA_test
					if not (args.save_model == ""):
						print("Saving model to {}".format(args.save_model))
						torch.save(
							{
								"epoch": k,
								"nepochs": args.nepochs,
								"nbatches": nbatches,
								"nbatches_test": nbatches_test,
								"iter": j + 1,
								"state_dict": dlrm.state_dict(),
								"train_acc": gA,
								"train_loss": gL,
								"test_acc": gA_test,
								"test_loss": gL_test,
								"total_loss": total_loss,
								"total_accu": total_accu,
								"opt_state_dict": optimizer.state_dict(),
							},
							args.save_model,
						)

				if args.mlperf_logging:
					is_best = validation_results['roc_auc'] > best_auc_test
					if is_best:
						best_auc_test = validation_results['roc_auc']

					print("Test_Iteration ", j + 1)
					print("Total_Iterations ", nbatches)
					print("Epoch ", k)
					print("Test_Loss ", validation_results['loss'])
					print("Test_recall ", validation_results['recall'])
					print("Test_precision ", validation_results['precision'])
					print("Test_f1 ", validation_results['f1'])
					print("Test_ap ", validation_results['ap'])
					print("Test_auc ", validation_results['roc_auc'])
					print("Best_auc ", best_auc_test)
					print("Test_Accuracy ", validation_results['accuracy'] * 100)
					print("Best_Accuracy ", best_gA_test * 100)
					print("\n")

					
				else:
					print("Test_Iteration ", j + 1)
					print("Total_Iterations ", nbatches)
					print("Test_Loss ", gL_test)
					print("Test_Accuracy ", gA_test * 100)
					print("Best_test_Accuracy ", best_gA_test * 100)
					print("\n")
					
				# Uncomment the line below to print out the total time with overhead
				# print("Total test time for this group: {}" \
				# .format(time_wrap(use_gpu) - accum_test_time_begin))

				if (args.mlperf_logging
					and (args.mlperf_acc_threshold > 0)
					and (best_gA_test > args.mlperf_acc_threshold)):
					print("MLPerf testing accuracy threshold "
						  + str(args.mlperf_acc_threshold)
						  + " reached, stop training")
					stop = 1
					break

				if (args.mlperf_logging
					and (args.mlperf_auc_threshold > 0)
					and (best_auc_test > args.mlperf_auc_threshold)):
					print("MLPerf testing auc threshold "
						  + str(args.mlperf_auc_threshold)
						  + " reached, stop training")
					break
					
				print("Cold_Test_Loss_" + str(j + 1) + " : " + str(gL_test))
				#print("Test_Accuracy" , str(j + 1) + " : " + str(gA_test * 100))
				print("Cold_Best_Test_Accuracy_" + str(j + 1) + " : " + str(best_gA_test * 100))
				print("Cold_Test_Data : ", data)
				#print("Cold_Testing_Time_Sec_"+ j + 1 " : " + (t2_test- t1_test ))
				
				
				print(
					"Testing_at_{}/{}_of_epoch_{}_".format(j + 1, nbatches, 0)
					+ "_loss_{:.6f}_accuracy_{:3.3f}%_best_{:3.3f}%".format(
						gL_test, gA_test * 100, best_gA_test * 100
					)
				)
				

		# At the end of train_normal_ld last iteration start training with train_hot_ld data 
		# ======================= Updating the hot_emb_l with emb_l =====================
		
		begin_emb_update = time_wrap(use_gpu)
		if stop == 0:
			for _, emb_dict in enumerate(hot_emb_dict):
				for _, (emb_no, emb_row) in enumerate(emb_dict):
					hot_row = emb_dict[(emb_no, emb_row)]
					data = dlrm.emb_l[emb_no].weight.data[emb_row]
					dlrm.hot_emb_l[0].weight.data[hot_row] = data

		end_emb_update = time_wrap(use_gpu)

		hot_emb_update += (end_emb_update - begin_emb_update)
		print("\nEMB_hot_Update :", (end_emb_update - begin_emb_update))
		print("\n") 
		



		begin_hot_training_loop = time_wrap(use_gpu)
		# we want to warm up the hot embedding table 
		# start_mb = 1000 and ending_minibatch = 2000
		if stop == 0:
			# =================================== set up done !! =========================================
			# Using Hot Train Data
			for j, (X, lS_o, lS_i, T) in enumerate(train_hot_ld):

				if j > ending_minibatch:
					break

				data = "hot" 
				if j < skip_upto_batch:
					continue

				t1 = time_wrap(use_gpu)

				# early exit if nbatches was set by the user and has been exceeded
				if nbatches_hot > 0 and j >= nbatches_hot:
					break
						
				should_print = ((j + 1) % args.print_freq == 0) or (j + 1 == nbatches_hot)

				begin_forward = time_wrap(use_gpu)
				# forward pass
				
				Z = dlrm_wrap(X, lS_o, lS_i, use_gpu, device, data)

				end_forward = time_wrap(use_gpu)

				# loss
				E = loss_fn_wrap(Z, T, use_gpu, device)
		
				# compute loss and accuracy
				L = E.detach().cpu().numpy()  # numpy array
				S = Z.detach().cpu().numpy()  # numpy array
				T = T.detach().cpu().numpy()  # numpy array
				mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
				A = np.sum((np.round(S, 0) == T).astype(np.uint8))


					
				# ================ First Whole Hot Embedding Screenshot ====================
				if j == args.starting_minibatch:
					begin_gpu_snapshot = time_wrap(use_gpu)
					prev_emb_hot =  copy.deepcopy(dlrm.hot_emb_l[0].weight)                 # making a copy on GPU - snapshop of the whole hot embedding 
					end_gpu_snapshot = time_wrap(use_gpu)
					gpu_operation_time = end_gpu_snapshot - begin_gpu_snapshot
				# ================ Fisrt Whole Hot Embedding Screenshot ====================
				
				



				if not args.inference_only:
					# scaled error gradient propagation
					# (where we do not accumulate gradients across mini-batches)
					optimizer.zero_grad()

					# backward pass
					E.backward()

					end_backward = time_wrap(use_gpu)

					# optimizer
					optimizer.step()

					end_optimizing = time_wrap(use_gpu)

					lr_scheduler.step()

					end_scheduling = time_wrap(use_gpu)


				
				# ================ Second Whole Hot Embedding Screenshot ====================
				if j == ending_minibatch:
					begin_gpu_snapshot = time_wrap(use_gpu)
					after_emb_hot =  copy.deepcopy(dlrm.hot_emb_l[0].weight)
					end_gpu_snapshot = time_wrap(use_gpu)
					gpu_operation_time = end_gpu_snapshot - begin_gpu_snapshot
				# ================ Second Whole Hot Embedding Screenshot ==================== 
				
				

				t2 = time_wrap(use_gpu)
				total_time += t2 - t1

				total_accu += A
				total_loss += L * mbs
				total_iter += 1
				total_samp += mbs

				forward_sampling_time += end_forward - begin_forward
				backward_sampling_time += end_backward - end_forward
				optimizer_sampling_time += end_optimizing - end_backward
				scheduler_sampling_time += end_scheduling - end_optimizing
				
				should_print = ((j + 1) % args.print_freq == 0) or (j + 1 == nbatches)
				
				should_test = (
					(args.test_freq > 0)
					and (args.data_generation == "dataset")
					and (((j + 1) % args.test_freq == 0) or (j + 1 == nbatches_hot))
				)

				# print time, loss and accuracy
				if should_print or should_test:
					gT = 1000.0 * total_time / total_iter if args.print_time else -1
					total_time = 0


					gA = total_accu / total_samp
					total_accu = 0


					gL = total_loss / total_samp
					total_loss = 0

					gForward = 1000 * forward_time / total_iter
					gBackward = 1000 * backward_time / total_iter
					gOptimizer = 1000 * optimizer_time / total_iter
					gScheduler = 1000 * scheduler_time /total_iter


					str_run_type = "inference" if args.inference_only else "training"


					print("Sampling_Loss_" + str(j + 1), ":", gL)
					print("Sampling_Accuracy_" + str(j + 1), ":", gA*100)
					print("Sampling_Train_data : ", data)

					#yass testing accuracy and loss 
					total_iter = 0
					total_samp = 0

					fwd_itr = 0
					bwd_itr = 0
					opt_itr = 0
					sch_itr = 0


				# testing
				if should_test and not args.inference_only:
					# ======================= Updating the emb_l with hot_emb_l =====================
					begin_emb_update = time_wrap(use_gpu)

					hot_emb = dlrm.hot_emb_l[0].weight.detach().cpu().numpy()
						
					for _, emb_dict in enumerate(hot_emb_dict):
						for _, (emb_no, emb_row) in enumerate(emb_dict):
							hot_row = emb_dict[(emb_no, emb_row)]
							data = torch.tensor(hot_emb[hot_row])
							dlrm.emb_l[emb_no].weight.data[emb_row] = data

					end_emb_update = time_wrap(use_gpu)

					cold_emb_update += (end_emb_update - begin_emb_update)
					print("\nEMB_normal_Update : ", (end_emb_update - begin_emb_update))
					print("\n")
						
					# ===============================================================================
				
					test_accu = 0
					test_loss = 0
					test_samp = 0

					if args.mlperf_logging:
						scores = []
						targets = []
					
					for i, (X_test, lS_o_test, lS_i_test, T_test) in enumerate(test_ld):
						data = "test"
						# early exit if nbatches was set by the user and was exceeded
						if nbatches > 0 and i >= nbatches:
							break

						should_print = 0
						# forward pass
						Z_test = dlrm_wrap(X_test, lS_o_test, lS_i_test, use_gpu, device, data)

						if args.mlperf_logging:
							S_test = Z_test.detach().cpu().numpy()  # numpy array
							T_test = T_test.detach().cpu().numpy()  # numpy array

							scores.append(S_test)
							targets.append(T_test)

						else:
							# loss
							E_test = loss_fn_wrap(Z_test, T_test, use_gpu, device)

							# compute loss and accuracy
							L_test = E_test.detach().cpu().numpy()  # numpy array
							S_test = Z_test.detach().cpu().numpy()  # numpy array
							T_test = T_test.detach().cpu().numpy()  # numpy array
							mbs_test = T_test.shape[0]  # = mini_batch_size except last
							A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))
							test_accu += A_test
							test_loss += L_test * mbs_test
							test_samp += mbs_test
							
					if args.mlperf_logging:

						scores = np.concatenate(scores, axis=0)
						targets = np.concatenate(targets, axis=0)


						metrics = {
							'loss' : sklearn.metrics.log_loss,
							'recall' : lambda y_true, y_score:
							sklearn.metrics.recall_score(
								y_true=y_true,
								y_pred=np.round(y_score)
							),
							'precision' : lambda y_true, y_score:
							sklearn.metrics.precision_score(
								y_true=y_true,
								y_pred=np.round(y_score)
							),
							'f1' : lambda y_true, y_score:
							sklearn.metrics.f1_score(
								y_true=y_true,
								y_pred=np.round(y_score)
							),
							'ap' : sklearn.metrics.average_precision_score,
							'roc_auc' : sklearn.metrics.roc_auc_score,
							'accuracy' : lambda y_true, y_score:
							sklearn.metrics.accuracy_score(
								y_true=y_true,
								y_pred=np.round(y_score)
							),
							# 'pre_curve' : sklearn.metrics.precision_recall_curve,
							# 'roc_curve' :  sklearn.metrics.roc_curve,
						}

						# print("Compute time for validation metric : ", end="")
						# first_it = True
						validation_results = {}
						for metric_name, metric_function in metrics.items():
							# if first_it:
							#     first_it = False
							# else:
							#     print(", ", end="")
							# metric_compute_start = time_wrap(False)
							validation_results[metric_name] = metric_function(
								targets,
								scores
							)
							# metric_compute_end = time_wrap(False)
							# met_time = metric_compute_end - metric_compute_start
							# print("{} {:.4f}".format(metric_name, 1000 * (met_time)),
							#      end="")
						# print(" ms")
						gA_test = validation_results['accuracy']
						gL_test = validation_results['loss']
					else:
						gA_test = test_accu / test_samp
						gL_test = test_loss / test_samp

					is_best = gA_test > best_gA_test
					if is_best:
						best_gA_test = gA_test
						if not (args.save_model == ""):
							print("Saving model to {}".format(args.save_model))
							torch.save(
								{
									"epoch": k,
									"nepochs": args.nepochs,
									"nbatches": nbatches,
									"nbatches_test": nbatches_test,
									"iter": j + 1,
									"state_dict": dlrm.state_dict(),
									"train_acc": gA,
									"train_loss": gL,
									"test_acc": gA_test,
									"test_loss": gL_test,
									"total_loss": total_loss,
									"total_accu": total_accu,
									"opt_state_dict": optimizer.state_dict(),
								},
								args.save_model,
							)

					if args.mlperf_logging:
						is_best = validation_results['roc_auc'] > best_auc_test
						if is_best:
							best_auc_test = validation_results['roc_auc']

						print("Test_Iteration ", j + 1)
						print("Total_Iterations ", nbatches)
						print("Epoch ", k)
						print("Test_Loss ", validation_results['loss'])
						print("Test_recall ", validation_results['recall'])
						print("Test_precision ", validation_results['precision'])
						print("Test_f1 ", validation_results['f1'])
						print("Test_ap ", validation_results['ap'])
						print("Test_auc ", validation_results['roc_auc'])
						print("Best_auc ", best_auc_test)
						print("Test_Accuracy ", validation_results['accuracy'] * 100)
						print("Best_Accuracy ", best_gA_test * 100)
						print("\n")

						
					else:
						print("Test_Iteration ", j + 1)
						print("Total_Iterations ", nbatches)
						print("Test_Loss ", gL_test)
						print("Test_Accuracy ", gA_test * 100)
						print("Best_test_Accuracy ", best_gA_test * 100)
						print("\n")
						
					# Uncomment the line below to print out the total time with overhead
					# print("Total test time for this group: {}" \
					# .format(time_wrap(use_gpu) - accum_test_time_begin))

					if (args.mlperf_logging
						and (args.mlperf_acc_threshold > 0)
						and (best_gA_test > args.mlperf_acc_threshold)):
						print("MLPerf testing accuracy threshold "
							  + str(args.mlperf_acc_threshold)
							  + " reached, stop training")
						stop = 1
						break

					if (args.mlperf_logging
						and (args.mlperf_auc_threshold > 0)
						and (best_auc_test > args.mlperf_auc_threshold)):
						print("MLPerf testing auc threshold "
							  + str(args.mlperf_auc_threshold)
							  + " reached, stop training")
						break

					print("Sampling_Test_Loss_" + str(j + 1)+ " : "+ str(gL_test))
					#print("Sampling_Test_Accuracy_"  + str(j + 1)+ " : "+ str(gA_test * 100))
					print("Sampling_Best_test_Accuracy_" + str(j + 1)+ " : "+ str(best_gA_test * 100))
					print("Sampling_Test_data :", data)
					print("\n")

		# ===================== END of warming up hot embedding tables ========================= #
		






		# =================================== Finding THE Threshold !! ========================================= #
		lower_drop_percentage = args.target_drop_percentage - 0.08					
		upper_drop_percentage = args.target_drop_percentage + 0.08


		# drop % is at minibatch level
		hot_entry_samples = int(args.sample_rate * len(train_hot))
		print("Number_Samples_Threshold_Setting : ",hot_entry_samples)
		sampled_train_data = np.random.randint(args.mini_batch_size * ending_minibatch, len(train_hot), size = hot_entry_samples )
		random_hot_emb = []
		
		for i, idx in enumerate(sampled_train_data):
			random_hot_emb.append(train_hot[idx])


		input_counter = 0
		total_threshold_setting_time = 0
		threshold_upper = 0.1 
		threshold_lower = 0
		final_threshold = 0
		
		if stop == 0:
			# Using Hot Train Data
			for minibatch in range(0,25):
				not_changing_inputs = 0
				begin_threshold = time_wrap(use_gpu)

				result_total = abs(prev_emb_hot - after_emb_hot) < (threshold_upper + threshold_lower)/2 
				final_total = torch.all(result_total, dim=1)
				final_total = final_total.int()
				final_total_number = torch.sum(final_total)
				final_total = final_total.detach().cpu().numpy() 

				for j, hot_tuple in enumerate(random_hot_emb):
					input_counter = 0
					for i, index in enumerate(hot_tuple[1]):
						if final_total[int(index)] == 1:
							input_counter = input_counter + 1
							
					if input_counter >= args.non_changing_index:
						not_changing_inputs = not_changing_inputs + 1
				

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
				# adding the theshold setting time per each minibatch 
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
		print("Overall_CPU_Time_Accelerated : ", end_cpu_operation - begin_cpu_operation )
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

		train_hot_ld_cluster_0 = dp.load_criteo_preprocessed_data_and_loaders_one(args, cluster_0)
		train_hot_ld_cluster_1 = dp.load_criteo_preprocessed_data_and_loaders_one(args, cluster_1)

		nbatches_hot_cluster_1 = args.num_batches if args.num_batches > 0 else len(train_hot_ld_cluster_1)
		nbatches_hot_cluster_0 = args.num_batches if args.num_batches > 0 else len(train_hot_ld_cluster_0)
		shm_0.close()   
		shm_1.close()
		shm_0.unlink() 
		shm_1.unlink()

		#================================= DONE WITH SlipStream Threshold ================================ #









		
		begin_hot_training_loop = time_wrap(use_gpu)
		#======== Using Hot Train Data -- with non-skippable entries that are changing =========== 
		if stop == 0:
			for j, (X, lS_o, lS_i, T) in enumerate(train_hot_ld_cluster_0):
				data = "hot"

				if j < skip_upto_batch:
					continue

				if args.mlperf_logging:
					scores = []
					targets = []

				# early exit if nbatches was set by the user and has been exceeded
				if nbatches_hot_cluster_0 > 0 and j >= nbatches_hot_cluster_0:
					break
						
				should_print = ((j + 1) % args.print_freq == 0) or (j + 1 == nbatches_hot)

				begin_forward = time_wrap(use_gpu)
				# forward pass
				Z = dlrm_wrap(X, lS_o, lS_i, use_gpu, device, data)
				end_forward = time_wrap(use_gpu)

				E = loss_fn_wrap(Z, T, use_gpu, device)

				# compute loss and accuracy
				L = E.detach().cpu().numpy()  # numpy array
				S = Z.detach().cpu().numpy()  # numpy array
				T = T.detach().cpu().numpy()  # numpy array
				mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
				A = np.sum((np.round(S, 0) == T).astype(np.uint8))

				if not args.inference_only:
					# scaled error gradient propagation
					# (where we do not accumulate gradients across mini-batches)
					optimizer.zero_grad()
					# backward pass
					E.backward()

					end_backward = time_wrap(use_gpu)

					# optimizer
					optimizer.step()

					end_optimizing = time_wrap(use_gpu)

					lr_scheduler.step()

					end_scheduling = time_wrap(use_gpu)


				total_accu += A
				total_loss += L * mbs
				total_iter += 1
				total_samp += mbs

				forward_hot_time += end_forward - begin_forward
				backward_hot_time += end_backward - end_forward
				optimizer_hot_time += end_optimizing - end_backward
				scheduler_hot_time += end_scheduling - end_optimizing



				should_print = ((j + 1) % args.print_freq == 0) or (j + 1 == nbatches)
				
				should_test = (
					(args.test_freq > 0)
					and (args.data_generation == "dataset")
					and (((j + 1) % args.test_freq == 0) or (j + 1 == nbatches_hot))
				)

				# print time, loss and accuracy
				if should_print or should_test:
					gT = 1000.0 * total_time / total_iter if args.print_time else -1
	 
					total_time = 0


					gA = total_accu / total_samp
					total_accu = 0


					gL = total_loss / total_samp
					total_loss = 0

					gForward = 1000 * forward_time / total_iter
					gBackward = 1000 * backward_time / total_iter
					gOptimizer = 1000 * optimizer_time / total_iter
					gScheduler = 1000 * scheduler_time /total_iter


					str_run_type = "inference" if args.inference_only else "training"

					print("Hot_Loss_" + str(j + 1)+ " : "+ str(gL))
					print("Hot_Train_Accuracy_"+ str(j + 1)+ " : "+ str(gA*100))
					print("Hot_Train_Data_" + str(j + 1)+ " : "+ str(data))


					#yass testing accuracy and loss 

					total_iter = 0
					total_samp = 0
					fwd_itr = 0
					bwd_itr = 0
					opt_itr = 0
					sch_itr = 0



				# testing
				if should_test and not args.inference_only:

					# Before testing the emb_l using hot_emb_l
					# ======================= Updating the emb_l with hot_emb_l =====================
						
					begin_emb_update = time_wrap(use_gpu)

					hot_emb = dlrm.hot_emb_l[0].weight.detach().cpu().numpy()
						
					for _, emb_dict in enumerate(hot_emb_dict):
						for _, (emb_no, emb_row) in enumerate(emb_dict):
							hot_row = emb_dict[(emb_no, emb_row)]
							data = torch.tensor(hot_emb[hot_row])
							dlrm.emb_l[emb_no].weight.data[emb_row] = data

					end_emb_update = time_wrap(use_gpu)
					cold_emb_update += (end_emb_update - begin_emb_update)
					print("\nEMB_normal_Update :", (end_emb_update - begin_emb_update))            
					print("\n")
						
					# ===============================================================================
				
					test_accu = 0
					test_loss = 0
					test_samp = 0

					if args.mlperf_logging:
						scores = []
						targets = []

					accum_test_time_begin = time_wrap(use_gpu)

					for i, (X_test, lS_o_test, lS_i_test, T_test) in enumerate(test_ld):
						data = "test"
						# early exit if nbatches was set by the user and was exceeded
						if nbatches > 0 and i >= nbatches:
							break

						should_print = 0
						# forward pass
						t1_test = time_wrap(use_gpu)
						Z_test = dlrm_wrap(X_test, lS_o_test, lS_i_test, use_gpu, device, data)
						t2_test = time_wrap(use_gpu)
						test_time_hot += (t2_test - t1_test)


						if args.mlperf_logging:
							S_test = Z_test.detach().cpu().numpy()  # numpy array
							T_test = T_test.detach().cpu().numpy()  # numpy array

							scores.append(S_test)
							targets.append(T_test)

						else:
							# loss
							E_test = loss_fn_wrap(Z_test, T_test, use_gpu, device)

							# compute loss and accuracy
							L_test = E_test.detach().cpu().numpy()  # numpy array
							S_test = Z_test.detach().cpu().numpy()  # numpy array
							T_test = T_test.detach().cpu().numpy()  # numpy array
							mbs_test = T_test.shape[0]  # = mini_batch_size except last
							A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))
							test_accu += A_test
							test_loss += L_test * mbs_test
							test_samp += mbs_test
							

					print("Hot_Test_Loss_"+ str(j + 1)+ " : "+ str(gL_test))
					#print("Test_Accuracy", str(j + 1)+ " : "+ str(gA_test * 100))
					print("Hot_Best_Test_Accuracy_" + str(j + 1)+ " : "+ str(best_gA_test * 100))
					print("Hot_Test_Data_" + str(j + 1)+ " : "+ str(data))
					print("\n")

					if args.mlperf_logging:
						scores = np.concatenate(scores, axis=0)
						targets = np.concatenate(targets, axis=0)

						metrics = {
							'loss' : sklearn.metrics.log_loss,
							'recall' : lambda y_true, y_score:
							sklearn.metrics.recall_score(
								y_true=y_true,
								y_pred=np.round(y_score)
							),
							'precision' : lambda y_true, y_score:
							sklearn.metrics.precision_score(
								y_true=y_true,
								y_pred=np.round(y_score)
							),
							'f1' : lambda y_true, y_score:
							sklearn.metrics.f1_score(
								y_true=y_true,
								y_pred=np.round(y_score)
							),
							'ap' : sklearn.metrics.average_precision_score,
							'roc_auc' : sklearn.metrics.roc_auc_score,
							'accuracy' : lambda y_true, y_score:
							sklearn.metrics.accuracy_score(
								y_true=y_true,
								y_pred=np.round(y_score)
							),
							# 'pre_curve' : sklearn.metrics.precision_recall_curve,
							# 'roc_curve' :  sklearn.metrics.roc_curve,
						}

						# print("Compute time for validation metric : ", end="")
						# first_it = True
						validation_results = {}
						for metric_name, metric_function in metrics.items():
							# if first_it:
							#     first_it = False
							# else:
							#     print(", ", end="")
							# metric_compute_start = time_wrap(False)
							validation_results[metric_name] = metric_function(
								targets,
								scores
							)
							# metric_compute_end = time_wrap(False)
							# met_time = metric_compute_end - metric_compute_start
							# print("{} {:.4f}".format(metric_name, 1000 * (met_time)),
							#      end="")
						# print(" ms")
						gA_test = validation_results['accuracy']
						gL_test = validation_results['loss']
					else:
						gA_test = test_accu / test_samp
						gL_test = test_loss / test_samp

					is_best = gA_test > best_gA_test
					if is_best:
						best_gA_test = gA_test
						if not (args.save_model == ""):
							print("Saving model to {}".format(args.save_model))
							torch.save(
								{
									"epoch": k,
									"nepochs": args.nepochs,
									"nbatches": nbatches,
									"nbatches_test": nbatches_test,
									"iter": j + 1,
									"state_dict": dlrm.state_dict(),
									"train_acc": gA,
									"train_loss": gL,
									"test_acc": gA_test,
									"test_loss": gL_test,
									"total_loss": total_loss,
									"total_accu": total_accu,
									"opt_state_dict": optimizer.state_dict(),
								},
								args.save_model,
							)

					if args.mlperf_logging:
						is_best = validation_results['roc_auc'] > best_auc_test
						if is_best:
							best_auc_test = validation_results['roc_auc']

						print("Test_Iteration ", j + 1)
						print("Total_Iterations ", nbatches)
						print("Epoch ", k)
						print("Test_Loss ", validation_results['loss'])
						print("Test_recall ", validation_results['recall'])
						print("Test_precision ", validation_results['precision'])
						print("Test_f1 ", validation_results['f1'])
						print("Test_ap ", validation_results['ap'])
						print("Test_auc ", validation_results['roc_auc'])
						print("Best_auc ", best_auc_test)
						print("Test_Accuracy ", validation_results['accuracy'] * 100)
						print("Best_Accuracy ", best_gA_test * 100)
						print("\n")

						
					else:
						print("Test_Iteration ", j + 1)
						print("Total_Iterations ", nbatches)
						print("Test_Loss ", gL_test)
						print("Test_Accuracy ", gA_test * 100)
						print("Best_test_Accuracy ", best_gA_test * 100)
						print("\n")
						
					# Uncomment the line below to print out the total time with overhead
					# print("Total test time for this group: {}" \
					# .format(time_wrap(use_gpu) - accum_test_time_begin))

					if (args.mlperf_logging
						and (args.mlperf_acc_threshold > 0)
						and (best_gA_test > args.mlperf_acc_threshold)):
						print("MLPerf testing accuracy threshold "
							  + str(args.mlperf_acc_threshold)
							  + " reached, stop training")
						stop = 1
						break

					if (args.mlperf_logging
						and (args.mlperf_auc_threshold > 0)
						and (best_auc_test > args.mlperf_auc_threshold)):
						print("MLPerf testing auc threshold "
							  + str(args.mlperf_auc_threshold)
							  + " reached, stop training")
						break

			# At the end of train_hot_ld last iteration update the emb_l
			# ======================= Updating the emb_l with hot_emb_l =====================
						
			begin_emb_update = time_wrap(use_gpu)
			hot_emb = dlrm.hot_emb_l[0].weight.detach().cpu().numpy()
						
			for _, emb_dict in enumerate(hot_emb_dict):
				for _, (emb_no, emb_row) in enumerate(emb_dict):
					hot_row = emb_dict[(emb_no, emb_row)]
					data = torch.tensor(hot_emb[hot_row])
					dlrm.emb_l[emb_no].weight.data[emb_row] = data

			end_emb_update = time_wrap(use_gpu)
			end_hot_training_loop = time_wrap(use_gpu)
			cold_emb_update += (end_emb_update - begin_emb_update)
			print("\nEMB_normal_Update :", (end_emb_update - begin_emb_update))
			print("\n")




		

		for j, (X, lS_o, lS_i, T) in enumerate(train_hot_ld_cluster_1):

			data = "hot"

			if j < skip_upto_batch:
				continue

			if args.mlperf_logging:
				scores = []
				targets = []

			# early exit if nbatches was set by the user and has been exceeded
			if nbatches_hot_cluster_0 > 0 and j >= nbatches_hot_cluster_0:
				break
					
			should_print = ((j + 1) % args.print_freq == 0) or (j + 1 == nbatches_hot)

			'''
			begin_forward = time_wrap(use_gpu)
			# forward pass
			Z = dlrm_wrap(X, lS_o, lS_i, use_gpu, device, data)
			end_forward = time_wrap(use_gpu)

			E = loss_fn_wrap(Z, T, use_gpu, device)

			# compute loss and accuracy
			L = E.detach().cpu().numpy()  # numpy array
			S = Z.detach().cpu().numpy()  # numpy array
			T = T.detach().cpu().numpy()  # numpy array
			mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
			A = np.sum((np.round(S, 0) == T).astype(np.uint8))

			if not args.inference_only:
				# scaled error gradient propagation
				# (where we do not accumulate gradients across mini-batches)
				optimizer.zero_grad()
				# backward pass
				E.backward()

				end_backward = time_wrap(use_gpu)

				# optimizer
				optimizer.step()

				end_optimizing = time_wrap(use_gpu)

				lr_scheduler.step()

				end_scheduling = time_wrap(use_gpu)


			total_accu += A
			total_loss += L * mbs
			total_iter += 1
			total_samp += mbs

			forward_hot_time += end_forward - begin_forward
			backward_hot_time += end_backward - end_forward
			optimizer_hot_time += end_optimizing - end_backward
			scheduler_hot_time += end_scheduling - end_optimizing

			'''

			should_print = ((j + 1) % args.print_freq == 0) or (j + 1 == nbatches)
			
			should_test = (
				(args.test_freq > 0)
				and (args.data_generation == "dataset")
				and (((j + 1) % args.test_freq == 0) or (j + 1 == nbatches_hot))
			)

			'''
			# print time, loss and accuracy
			if should_print or should_test:
				gT = 1000.0 * total_time / total_iter if args.print_time else -1
 
				total_time = 0


				gA = total_accu / total_samp
				total_accu = 0


				gL = total_loss / total_samp
				total_loss = 0

				gForward = 1000 * forward_time / total_iter
				gBackward = 1000 * backward_time / total_iter
				gOptimizer = 1000 * optimizer_time / total_iter
				gScheduler = 1000 * scheduler_time /total_iter


				str_run_type = "inference" if args.inference_only else "training"

				print("Hot_Loss_" + str(j + 1)+ " : "+ str(gL))
				print("Hot_Train_Accuracy_"+ str(j + 1)+ " : "+ str(gA*100))
				print("Hot_Train_Data_" + str(j + 1)+ " : "+ str(data))


				#yass testing accuracy and loss 

				total_iter = 0
				total_samp = 0
				fwd_itr = 0
				bwd_itr = 0
				opt_itr = 0
				sch_itr = 0

			'''

			# testing
			if should_test and not args.inference_only:
					
				# ===============================================================================
			
				test_accu = 0
				test_loss = 0
				test_samp = 0

				if args.mlperf_logging:
					scores = []
					targets = []

				accum_test_time_begin = time_wrap(use_gpu)

				for i, (X_test, lS_o_test, lS_i_test, T_test) in enumerate(test_ld):
					data = "test"
					# early exit if nbatches was set by the user and was exceeded
					if nbatches > 0 and i >= nbatches:
						break

					should_print = 0
					# forward pass
					t1_test = time_wrap(use_gpu)
					Z_test = dlrm_wrap(X_test, lS_o_test, lS_i_test, use_gpu, device, data)
					t2_test = time_wrap(use_gpu)
					test_time_hot += (t2_test - t1_test)


					if args.mlperf_logging:
						S_test = Z_test.detach().cpu().numpy()  # numpy array
						T_test = T_test.detach().cpu().numpy()  # numpy array

						scores.append(S_test)
						targets.append(T_test)

					else:
						# loss
						E_test = loss_fn_wrap(Z_test, T_test, use_gpu, device)

						# compute loss and accuracy
						L_test = E_test.detach().cpu().numpy()  # numpy array
						S_test = Z_test.detach().cpu().numpy()  # numpy array
						T_test = T_test.detach().cpu().numpy()  # numpy array
						mbs_test = T_test.shape[0]  # = mini_batch_size except last
						A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))
						test_accu += A_test
						test_loss += L_test * mbs_test
						test_samp += mbs_test
						

				print("Hot_Test_Loss_"+ str(j + 1)+ " : "+ str(gL_test))
				#print("Test_Accuracy", str(j + 1)+ " : "+ str(gA_test * 100))
				print("Hot_Best_Test_Accuracy_" + str(j + 1)+ " : "+ str(best_gA_test * 100))
				print("Hot_Test_Data_" + str(j + 1)+ " : "+ str(data))
				print("\n")

				if args.mlperf_logging:
					scores = np.concatenate(scores, axis=0)
					targets = np.concatenate(targets, axis=0)

					metrics = {
						'loss' : sklearn.metrics.log_loss,
						'recall' : lambda y_true, y_score:
						sklearn.metrics.recall_score(
							y_true=y_true,
							y_pred=np.round(y_score)
						),
						'precision' : lambda y_true, y_score:
						sklearn.metrics.precision_score(
							y_true=y_true,
							y_pred=np.round(y_score)
						),
						'f1' : lambda y_true, y_score:
						sklearn.metrics.f1_score(
							y_true=y_true,
							y_pred=np.round(y_score)
						),
						'ap' : sklearn.metrics.average_precision_score,
						'roc_auc' : sklearn.metrics.roc_auc_score,
						'accuracy' : lambda y_true, y_score:
						sklearn.metrics.accuracy_score(
							y_true=y_true,
							y_pred=np.round(y_score)
						),
						# 'pre_curve' : sklearn.metrics.precision_recall_curve,
						# 'roc_curve' :  sklearn.metrics.roc_curve,
					}

					# print("Compute time for validation metric : ", end="")
					# first_it = True
					validation_results = {}
					for metric_name, metric_function in metrics.items():
						# if first_it:
						#     first_it = False
						# else:
						#     print(", ", end="")
						# metric_compute_start = time_wrap(False)
						validation_results[metric_name] = metric_function(
							targets,
							scores
						)
						# metric_compute_end = time_wrap(False)
						# met_time = metric_compute_end - metric_compute_start
						# print("{} {:.4f}".format(metric_name, 1000 * (met_time)),
						#      end="")
					# print(" ms")
					gA_test = validation_results['accuracy']
					gL_test = validation_results['loss']
				else:
					gA_test = test_accu / test_samp
					gL_test = test_loss / test_samp

				is_best = gA_test > best_gA_test
				if is_best:
					best_gA_test = gA_test
					if not (args.save_model == ""):
						print("Saving model to {}".format(args.save_model))
						torch.save(
							{
								"epoch": k,
								"nepochs": args.nepochs,
								"nbatches": nbatches,
								"nbatches_test": nbatches_test,
								"iter": j + 1,
								"state_dict": dlrm.state_dict(),
								"train_acc": gA,
								"train_loss": gL,
								"test_acc": gA_test,
								"test_loss": gL_test,
								"total_loss": total_loss,
								"total_accu": total_accu,
								"opt_state_dict": optimizer.state_dict(),
							},
							args.save_model,
						)

				if args.mlperf_logging:
					is_best = validation_results['roc_auc'] > best_auc_test
					if is_best:
						best_auc_test = validation_results['roc_auc']

					print("Test_Iteration ", j + 1)
					print("Total_Iterations ", nbatches)
					print("Epoch ", k)
					print("Test_Loss ", validation_results['loss'])
					print("Test_recall ", validation_results['recall'])
					print("Test_precision ", validation_results['precision'])
					print("Test_f1 ", validation_results['f1'])
					print("Test_ap ", validation_results['ap'])
					print("Test_auc ", validation_results['roc_auc'])
					print("Best_auc ", best_auc_test)
					print("Test_Accuracy ", validation_results['accuracy'] * 100)
					print("Best_Accuracy ", best_gA_test * 100)
					print("\n")

					
				else:
					print("Test_Iteration ", j + 1)
					print("Total_Iterations ", nbatches)
					print("Test_Loss ", gL_test)
					print("Test_Accuracy ", gA_test * 100)
					print("Best_test_Accuracy ", best_gA_test * 100)
					print("\n")
					
				# Uncomment the line below to print out the total time with overhead
				# print("Total test time for this group: {}" \
				# .format(time_wrap(use_gpu) - accum_test_time_begin))

				if (args.mlperf_logging
					and (args.mlperf_acc_threshold > 0)
					and (best_gA_test > args.mlperf_acc_threshold)):
					print("MLPerf testing accuracy threshold "
						  + str(args.mlperf_acc_threshold)
						  + " reached, stop training")
					stop = 1
					break

				if (args.mlperf_logging
					and (args.mlperf_auc_threshold > 0)
					and (best_auc_test > args.mlperf_auc_threshold)):
					print("MLPerf testing auc threshold "
						  + str(args.mlperf_auc_threshold)
						  + " reached, stop training")
					break




		# ===============================================================================
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
		sampling_overall_time = forward_sampling_time + backward_sampling_time + optimizer_sampling_time + scheduler_sampling_time

		print("GPU_Operation_Time_Sec :", gpu_operation_time)
		print("Overall_Threshold_Quest_Time : ",total_threshold_setting_time)
		print("CPU_Operation_Time_Sec :", cpu_operation_time)
		slipstream_overall_overhead = gpu_operation_time + cpu_operation_time + total_threshold_setting_time 
		print("Overall_SlipStream_Time_Sec :", slipstream_overall_overhead)
		print("=====================================================================")
		print(" ")

		print("=====================================================================")
		print("Hot_Forward_Time_Sec :", forward_hot_time)
		print("Hot_Backward_Time_Sec :", backward_hot_time)
		print("Hot_Optimizer_Time_Sec :", optimizer_hot_time)
		print("Hot_Scheduler_Time_Sec :", scheduler_hot_time)
		hot_overall_training = forward_hot_time+backward_hot_time+optimizer_hot_time+scheduler_hot_time+sampling_overall_time
		print("Overall_Hot_Training_Time_Sec :",hot_overall_training)

		print(" ")
		print("Testing_Time_Hot_Sec :",test_time_hot)
		print("Overall_Testing_Training_Time_Hot_Sec :",test_time_hot+hot_overall_training)
		print("=====================================================================")
		print(" ")


		print("=====================================================================")
		overall_emb_update = cold_emb_update + hot_emb_update
		print("Normal_EMB_Update : ", cold_emb_update)
		print("Hot_EMB_Update : ",hot_emb_update)
		print("Overall_EMB_Update : ", overall_emb_update)
		print("=====================================================================")
		print(" ")

		overall_time =cold_overall_training + hot_overall_training + overall_emb_update + slipstream_overall_overhead 
		print("=====================================================================")
		print("Overall_Training_Time_Sec :",overall_time)
		print("Overall_Training_Testing_Time_Sec :",overall_time+ test_time_hot + test_time_cold )
		print("=====================================================================")
		print(" ")

		print("Slipstream_Overhead Percentage :",slipstream_overall_overhead/overall_time)

		k += 1  # nepochs


	# profiling
	if args.enable_profiling:
		with open("dlrm_s_pytorch.prof", "w") as prof_f:
			prof_f.write(prof.key_averages().table(sort_by="cpu_time_total"))
			prof.export_chrome_trace("./dlrm_s_pytorch.json")
		# print(prof.key_averages().table(sort_by="cpu_time_total"))

	# plot compute graph
	if args.plot_compute_graph:
		sys.exit(
			"ERROR: Please install pytorchviz package in order to use the"
			+ " visualization. Then, uncomment its import above as well as"
			+ " three lines below and run the code again."
		)
		# V = Z.mean() if args.inference_only else E
		# dot = make_dot(V, params=dict(dlrm.named_parameters()))
		# dot.render('dlrm_s_pytorch_graph') # write .pdf file

	# test prints
	if not args.inference_only and args.debug_mode:
		print("updated parameters (weights and bias):")
		for param in dlrm.parameters():
			print(param.detach().cpu().numpy())

	# export the model in onnx
	if args.save_onnx:
		dlrm_pytorch_onnx_file = "dlrm_s_pytorch.onnx"
		batch_size = X_onnx.shape[0]
		# debug prints
		# print("batch_size", batch_size)
		# print("inputs", X_onnx, lS_o_onnx, lS_i_onnx)
		# print("output", dlrm_wrap(X_onnx, lS_o_onnx, lS_i_onnx, use_gpu, device))

		# force list conversion
		# if torch.is_tensor(lS_o_onnx):
		#    lS_o_onnx = [lS_o_onnx[j] for j in range(len(lS_o_onnx))]
		# if torch.is_tensor(lS_i_onnx):
		#    lS_i_onnx = [lS_i_onnx[j] for j in range(len(lS_i_onnx))]
		# force tensor conversion
		# if isinstance(lS_o_onnx, list):
		#     lS_o_onnx = torch.stack(lS_o_onnx)
		# if isinstance(lS_i_onnx, list):
		#     lS_i_onnx = torch.stack(lS_i_onnx)
		# debug prints
		print("X_onnx.shape", X_onnx.shape)
		if torch.is_tensor(lS_o_onnx):
			print("lS_o_onnx.shape", lS_o_onnx.shape)
		else:
			for oo in lS_o_onnx:
				print("oo.shape", oo.shape)
		if torch.is_tensor(lS_i_onnx):
			print("lS_i_onnx.shape", lS_i_onnx.shape)
		else:
			for ii in lS_i_onnx:
				print("ii.shape", ii.shape)

		# name inputs and outputs
		o_inputs = ["offsets"] if torch.is_tensor(lS_o_onnx) else ["offsets_"+str(i) for i in range(len(lS_o_onnx))]
		i_inputs = ["indices"] if torch.is_tensor(lS_i_onnx) else ["indices_"+str(i) for i in range(len(lS_i_onnx))]
		all_inputs = ["dense_x"] + o_inputs + i_inputs
		#debug prints
		print("inputs", all_inputs)

		# create dynamic_axis dictionaries
		do_inputs = [{'offsets': {1 : 'batch_size' }}] if torch.is_tensor(lS_o_onnx) else [{"offsets_"+str(i) :{0 : 'batch_size'}} for i in range(len(lS_o_onnx))]
		di_inputs = [{'indices': {1 : 'batch_size' }}] if torch.is_tensor(lS_i_onnx) else [{"indices_"+str(i) :{0 : 'batch_size'}} for i in range(len(lS_i_onnx))]
		dynamic_axes = {'dense_x' : {0 : 'batch_size'}, 'pred' : {0 : 'batch_size'}}
		for do in do_inputs:
			dynamic_axes.update(do)
		for di in di_inputs:
			dynamic_axes.update(di)
		# debug prints
		print(dynamic_axes)

		# export model
		torch.onnx.export(
			dlrm, (X_onnx, lS_o_onnx, lS_i_onnx), dlrm_pytorch_onnx_file, verbose=True, use_external_data_format=True, opset_version=11, input_names=all_inputs, output_names=["pred"], dynamic_axes=dynamic_axes
		)
		# recover the model back
		dlrm_pytorch_onnx = onnx.load(dlrm_pytorch_onnx_file)
		# check the onnx model
		onnx.checker.check_model(dlrm_pytorch_onnx)
		'''
		
		# run model using onnxruntime
		import onnxruntime as rt

		dict_inputs = {}
		dict_inputs["dense_x"] = X_onnx.numpy().astype(np.float32)
		if torch.is_tensor(lS_o_onnx):
			dict_inputs["offsets"] = lS_o_onnx.numpy().astype(np.int64)
		else:
			for i in range(len(lS_o_onnx)):
				dict_inputs["offsets_"+str(i)] = lS_o_onnx[i].numpy().astype(np.int64)
		if torch.is_tensor(lS_i_onnx):
			dict_inputs["indices"] = lS_i_onnx.numpy().astype(np.int64)
		else:
			for i in range(len(lS_i_onnx)):
				dict_inputs["indices_"+str(i)] = lS_i_onnx[i].numpy().astype(np.int64)
		print("dict_inputs", dict_inputs)

		sess = rt.InferenceSession(dlrm_pytorch_onnx_file, rt.SessionOptions())
		prediction = sess.run(output_names=["pred"], input_feed=dict_inputs)
		print("prediction", prediction)
		'''
