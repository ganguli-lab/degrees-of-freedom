from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import logging
import jax.profiler

# Put the imports from the sub-functions here
from architectures import SimpleCNN, KerasResNets, WideResnet
from generate_data import setupMNIST, setupFashionMNIST, setupCIFAR10, setupCIFAR100, setupSVHN
from training_utils import generate_projection, flatten_leaves, theta_to_paramstree, sparse_theta_to_paramstree
from data_utils import save_obj, load_obj, sizeof_fmt
from logging_tools import loggingSetup, gitstatus, envstatus, rnginit

import matplotlib.pyplot as plt
import numpy as onp 
import jax.numpy as jnp 
from jax.tree_util import tree_multimap
from functools import partial
import math
import jax
import flax
from flax.training import lr_schedule
import tensorflow as tf

import time
from scipy import sparse



# Tree Utils 
def boolean_condition_to_float_on_tree(f,t):
	return tree_multimap(f,t)

# Should be careful: multiply_tree and multiply_trees are different
# Possibly rename?
def multiply_tree(factor,t):
	fn = lambda leaf: factor*leaf
	return tree_multimap(fn,t)

@partial(partial, tree_multimap)
def add_trees(t1,t2):
	return t1+t2

@partial(partial, tree_multimap)
def multiply_trees(t1,t2):
	return t1*t2

def flatten_tree_into_vector(t):
	return onp.concatenate([q.reshape([-1]) for q in jax.tree_leaves(t)])

@partial(partial, tree_multimap)
def zero_tree(t1):
	return 0.0

def linear_combine_trees(f1,t1,f2,t2):
	return add_trees(multiply_tree(f1,t1),multiply_tree(f2,t2))

# Subfunctions for training 
@jax.vmap
def cross_entropy_loss(logits, label):
	return -logits[label]

def normal_loss(params, batch, model_to_use, train=True):
	logits = jax.nn.log_softmax(model_to_use.call(params, batch['image'], train = train))
	loss = jnp.mean(cross_entropy_loss(logits, batch['label']))
	return loss

def normal_accuracy(params,batch, model_to_use, train=True):
	logits = jax.nn.log_softmax(model_to_use.call(params, batch['image'], train = train))
	return jnp.mean(jnp.argmax(logits, -1) == batch['label'])

@jax.jit
def train_step(step, optimizer, batch, state, lr):

	def loss_fn(model):

		with flax.nn.stateful(state) as new_state: #to update the state
			logits = model(batch['image'])

		loss = jnp.mean(cross_entropy_loss(
			logits, batch['label']))
		
		params = model.params

		#for L2 penalty on the loss
		weight_penalty_params = jax.tree_leaves(params)
		weight_decay = 1e-4
		weight_l2 = sum([jnp.sum(x ** 2)
						  for x in weight_penalty_params
						  if x.ndim > 2])
		weight_penalty = weight_decay * 0.5 * weight_l2

		return loss + weight_penalty, new_state

	grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

	(_, new_state), grad = grad_fn(optimizer.target)

	#new_optimizer = optimizer.apply_gradient(grad, learning_rate=learning_rate_fn(step))
	new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

	return new_optimizer, new_state


# Set up the arguements for main
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=3, help='training epochs per run')
parser.add_argument('--points_to_collect', type=int, default=1, help = 'runs per dimension')
#parser.add_argument('--lr', type=float, default=5e-2, help = 'learning rate')
parser.add_argument('--model', type=str, default='TinyCNN', help = 'model to train')
parser.add_argument('--dataset', type=str, default='MNIST', help = 'dataset for training')
parser.add_argument('--jit_grad', default=False, action='store_true', help = 'jit the gradient function for smaller projection matrices')


def main(args):
	## Load in args which set parameters for runs
	epochs = args.epochs
	points_to_collect = args.points_to_collect #number of repetitions per d
	#lr = args.lr
	model = args.model
	dataset = args.dataset
	jit_grad = args.jit_grad

	# Parameters for run
	N_runs = points_to_collect

	# the fractions of weights / (weights+biases) to keep in the mask
	fractions_tokeep_list = list(sorted([1.0,0.5,0.25,0.1,0.05,0.025,0.01,0.005,0.0025,0.001,0.0005,0.00025,0.0001]))

	to_mask_epochs = 2.0
	# Epochs parameter will be passed in by user
	#epochs = 6.0

	#rewinds to init for None, otherwise rewinds to the step specified
	steps_to_partially_pretrain = None
	# steps_to_partially_pretrain = 16

	# if you want to ignore biases in the lottery masks
	ignore_biases_bool = False

	# masks not related to the actual weights and biases
	randomize_masks = False

	default_LR = 5e-2
	LR_with_mask= 1e-1

	mass = 0.9


	# Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
	# it unavailable to JAX.
	tf.config.experimental.set_visible_devices([], "GPU")

	## Logging 
	# Logger specifications
	do_log = False
	do_gitchecks = False
	do_envchecks = False

	log_dir = '../lottery-subspace-data'
	param_str = '%s_%s' % (model, dataset)

	logger = logging.getLogger("my logger")
	scriptname = os.path.basename(__file__).rstrip('.py') # Get name of script
	aname, _ = loggingSetup(logger, scriptname, log_dir, do_log=do_log, param_str = param_str)
	result_file = '%s_results' % (aname)  # Outfile name

	# Print current environment and git status to the log
	if do_gitchecks:
		gitstatus(logger)

	if do_envchecks:
		envstatus(logger, use_gpu = True)


	# Start log with experimental parameters
	logger.info('\n ---Code Output---\n')
	logger.info('\n')
	logger.info('[Lottery Tickets] Magnitude pruning after training: \n')
	logger.info('\n')
	logger.info('Number of runs: %i \n' % N_runs)
	logger.info('Pruning Levels: %s \n' % str(fractions_tokeep_list))
	logger.info('Model: %s \n' % (model))
	logger.info('Dataset: %s \n' % (dataset))
	logger.info('Train for %i epochs with learning rate %.2f before masking. \n' % (to_mask_epochs, default_LR))
	logger.info('Train for %i epochs with starting learning rate %.2f after masking. \n' % (epochs, LR_with_mask))
	logger.info('[Ignore biases] %s [Randomize Mask] %s \n' % (str(ignore_biases_bool), str(randomize_masks)))
	logger.info('\n')

	batch_size = 128

	## Setup data
	if (dataset == 'MNIST'):
		x_train, full_train_dict, train_ds, test_ds, classes = setupMNIST()
		input_shape = (batch_size, 28, 28, 1)
	elif (dataset == 'fashionMNIST'):
		x_train, full_train_dict, train_ds, test_ds, classes = setupFashionMNIST()
		input_shape = (batch_size, 28, 28, 1)
	elif (dataset == 'SVHN'):
		x_train, full_train_dict, train_ds, test_ds, classes = setupSVHN()
		input_shape = (batch_size, 32, 32, 3)
	elif (dataset == 'cifar10'):
		x_train, full_train_dict, train_ds, test_ds, classes = setupCIFAR10()
		input_shape = (batch_size, 32, 32, 3)
	elif (dataset == 'cifar100'):
		x_train, full_train_dict, train_ds, test_ds, classes = setupCIFAR100()
		input_shape = (batch_size, 32, 32, 3)
	else:
		logging.error('Dataset not recognized \n')

	train_ds = full_train_dict
	test_ds_normalized = dict(test_ds)

	## Initialize model
	global net
	if (model == 'TinyCNN'):
		model_to_use = SimpleCNN.partial(
			channels = [16,32],
			classes = classes,
			)
	elif (model == 'SmallCNN'):
		model_to_use = SimpleCNN.partial(
			channels = [32,64,64],
			classes = classes,
			)
	elif (model == 'MediumCNN'):
		model_to_use = SimpleCNN.partial(
			channels = [32,64,64,128],
			classes = classes,
			)
	elif (model == 'ResNet_BNotf'):
		model_to_use = KerasResNets.partial(
			num_classes = classes,
			use_batch_norm = True,
		)
	elif (model == 'WideResNet'):
		model_to_use = WideResnet.partial(
			blocks_per_group=2,
			channel_multiplier=4,
			num_outputs=100,
			dropout_rate=0.0
		)
	else:
		logger.error('Model type not recognized\n')

	## Function that pretrains the network
	def get_pretrain(random_key_int, input_shape, to_mask_epochs=1, steps_to_partial_pretrain=None):

		params_now_partially_pretrained = None

		train_ds_touse = train_ds

		LR_schedule_now = []


		with flax.nn.stateful() as init_state_raw:
			_, initial_params = model_to_use.init_by_shape(jax.random.PRNGKey(random_key_int), [(input_shape, jnp.float32)])
			model = flax.nn.Model(model_to_use, initial_params)
			init_state = init_state_raw

		state = init_state # for the batch norm params

		optimizer = flax.optim.Momentum(learning_rate=0.1, beta=0.9).create(model) 

		total_it = -1  

		while True:

			epoch_now = total_it / (len(x_train)/float(batch_size))

			batch_ids = onp.random.choice(range(len(train_ds["image"])),(batch_size),replace = False)
			batch = {
				"image": train_ds["image"][batch_ids],
				"label": train_ds["label"][batch_ids],
			}

			total_it = total_it + 1

			if total_it / (len(x_train)/float(batch_size)) > to_mask_epochs:
				break

			#getting the update step
			optimizer,state = train_step(total_it, optimizer, batch, state, default_LR)

			if (steps_to_partial_pretrain is not None) and (steps_to_partial_pretrain==total_it+1):
				params_now_partially_pretrained = optimizer.target.params

		#getting the params out
		params_now = optimizer.target.params

		#doing the BN averaing on the test
		test_loss_out_testBN = normal_loss(params_now,test_ds, model_to_use)
		test_accuracy_out_testBN = normal_accuracy(params_now,test_ds, model_to_use)


		return initial_params, params_now, params_now_partially_pretrained

	def get_LR(step):
		if step < 400:
			return LR_with_mask
		elif step < 800:
			return 0.3*LR_with_mask
		elif step < 1200:
			return 0.1*LR_with_mask
		else:
			return 0.03*LR_with_mask
	# End of function

	inits_all = [[[] for _ in range(len(fractions_tokeep_list))] for _ in range(N_runs)]
	pretrains_all = [[[] for _ in range(len(fractions_tokeep_list))] for _ in range(N_runs)]
	partial_pretrains_all = [[[] for _ in range(len(fractions_tokeep_list))] for _ in range(N_runs)]
	masks_all = [[[] for _ in range(len(fractions_tokeep_list))] for _ in range(N_runs)]
	finals_all = [[[] for _ in range(len(fractions_tokeep_list))] for _ in range(N_runs)]

	fracs_on_list = []
	test_accs_list = []
	train_accs_list = []

	# Loop over runs
	for i_run in range(N_runs):
		params_init, params_pretrained, params_partially_pretrained = get_pretrain(
			23413278+i_run*len(fractions_tokeep_list),#+i_th,
			input_shape,
			to_mask_epochs,
			steps_to_partial_pretrain = steps_to_partially_pretrain,
			)

		params_sorted = onp.sort(onp.abs(flatten_tree_into_vector(params_pretrained)))[::-1]
		num_params = len(params_sorted)

		# Loop over pruning levels
		for i_th, size_th in enumerate(fractions_tokeep_list):
			
			#getting the performance
			outputs_now = model_to_use.call(params_pretrained,test_ds["image"][:500])
			pretrain_acc_now = onp.mean(onp.argmax(outputs_now,axis=-1) == test_ds["label"][:500])

			if randomize_masks == False:
				params_to_use_for_mask = params_pretrained
			else:
				params_to_use_for_mask = uniform_random_tree(params_pretrained)

			if size_th >= 1.0:
				threshold = 0.0
			else:
				threshold = params_sorted[onp.ceil(size_th*num_params).astype(int)]
			
			#getting the mask
			# The commented out part is for layerwise
			if ignore_biases_bool:
				bool_tree = boolean_condition_to_float_on_tree(
					lambda x: (jnp.abs(x) >= threshold) if len(x.shape)>1 else jnp.ones_like(x).astype(jnp.bool_),
					params_pretrained
				)
				# bool_tree = boolean_condition_to_float_on_tree(
				# 	lambda x: (jnp.abs(x) >= float(sorted(jnp.abs(x.reshape([-1])))[int((1.0-size_th)*len(x.reshape([-1])))])) if len(x.shape)>1 else jnp.ones_like(x).astype(jnp.bool_),
				# 	params_to_use_for_mask
				# )
			else:
				bool_tree = boolean_condition_to_float_on_tree(
					lambda x: (jnp.abs(x) >= threshold),
					params_pretrained
				)
				# bool_tree = boolean_condition_to_float_on_tree(
				# 	lambda x: (jnp.abs(x) >= float(sorted(jnp.abs(x.reshape([-1])))[int((1.0-size_th)*len(x.reshape([-1])))])),
				# 	params_to_use_for_mask
				# )
			#print('Created mask')


			masks_all[i_run][i_th] = bool_tree

			bool_tree_flat = flatten_tree_into_vector(bool_tree)

			#with mask
			outputs_now = model_to_use.call(multiply_trees(bool_tree, params_pretrained),test_ds["image"][:500])
			masked_pretrain_acc_now = onp.mean(onp.argmax(outputs_now,axis=-1) == test_ds["label"][:500])
			
			#with mask at init
			outputs_now = model_to_use.call(multiply_trees(bool_tree, params_init),test_ds["image"][:500])
			masked_init_acc_now = onp.mean(onp.argmax(outputs_now,axis=-1) == test_ds["label"][:500])

			logger.info("run="+str(i_run)+" threshold="+str(i_th)+" fraction="+str(onp.mean(bool_tree_flat))+
				" pretrain test="+str(pretrain_acc_now)+" w mask="+str(masked_pretrain_acc_now)+" init w mask="+str(masked_init_acc_now) + "\n")
			
			#getting a masked fn
			def make_masked_function(f_in,mask_in):
				@jax.jit
				def f_masked(params,inputs):
					masked_params = multiply_trees(mask_in,params)
					return f_in(masked_params,inputs)
				
				return f_masked

			masked_model_call = make_masked_function(model_to_use.call, bool_tree)

			@jax.jit
			def get_masked_loss(params_in,batch_in):
				logits_out = masked_model_call(params_in,batch_in["image"])
				loss_out = cross_entropy_loss(logits_out,batch_in["label"])
				return jnp.mean(loss_out)

			@jax.jit
			def get_masked_loss_and_grad(params_in,batch_in):
				return jax.value_and_grad(lambda ps: get_masked_loss(ps,batch_in))(params_in)

			#training with a mask
			ts = []

			momentum_vector = zero_tree(params_init) #just a zero-filled tree

			#back to initialization / partially pretrained
			if steps_to_partially_pretrain is None:
				params_now = params_init
			else:
				params_now = params_partially_pretrained
			  
			total_it = -1

			while True:

				epoch_now = total_it / (len(x_train)/float(batch_size))


				batch_ids = onp.random.choice(range(len(train_ds["image"])),(batch_size),replace = False)
				batch = {
					"image": train_ds["image"][batch_ids],
					"label": train_ds["label"][batch_ids],
				}

				total_it = total_it + 1

				if total_it / (len(x_train)/float(batch_size)) > epochs:
					break

				#getting the gradient
				if total_it == 0:
					previous_loss= None
				else:
					previous_loss = loss_out_now

				loss_out_now, grad_out_now = get_masked_loss_and_grad(params_now, batch)


				momentum_vector = linear_combine_trees(mass,momentum_vector,get_LR(total_it),grad_out_now)

				params_now = linear_combine_trees(1.0,params_now,-1.0,momentum_vector)

				if loss_out_now == previous_loss:
					logger.info("Nothing changing, breaking the run! \n")
					break

				if total_it % 100 == 0:
					outputs_now = masked_model_call(params_now,test_ds["image"][:500])
					acc_now = onp.mean(onp.argmax(outputs_now,axis=-1) == test_ds["label"][:500])
					logger.info("[Iteration %i]: Loss: %.3f, Acc: %.3f \n" % (total_it,loss_out_now,acc_now))

			#storing the final result (Note this is just saveing the model files)
			finals_all[i_run][i_th] = params_now	

			# Save out the accuracy
			frac_on = onp.mean(flatten_tree_into_vector(masks_all[i_run][i_th]))

			test_preds_now = masked_model_call(params_now,test_ds["image"])
			test_acc_now = onp.mean(onp.argmax(test_preds_now,axis=-1) == test_ds["label"])

			#train_preds_now = masked_model_call(params_now,train_ds["image"])
			#train_preds_now = get_preds(train_ds["image"],multiply_trees(masks_all[i_run][i_th],finals_all[i_run][i_th]),local_batch=5000)
			local_batch = 5000
			local_its = int(onp.ceil(float(len(train_ds["image"]))/float(local_batch)))
			train_preds_now = onp.zeros((len(train_ds["image"]), classes))
			for it in range(local_its):
				i1 = it*local_batch
				i2 = min([(it+1)*local_batch,len(train_ds["image"])])
				images_now = train_ds["image"][i1:i2]
				train_preds_now[i1:i2, :] = masked_model_call(params_now, images_now)
			train_acc_now = onp.mean(onp.argmax(train_preds_now,axis=-1) == train_ds["label"])

			fracs_on_list.append(frac_on)
			test_accs_list.append(test_acc_now) 
			train_accs_list.append(train_acc_now)   

			logger.info("Compression Ratio: %.2e, Train Acc: %.2f, Test Acc: %.2f \n" % (1.0/frac_on,train_acc_now,test_acc_now))

	fracs_on_np = onp.array(fracs_on_list).reshape([N_runs,-1])
	test_accs_np = onp.array(test_accs_list).reshape([N_runs,-1])
	train_accs_np = onp.array(train_accs_list).reshape([N_runs,-1])

	data_dict = {
		"test_accs_np": test_accs_np,
		"train_accs_np": train_accs_np,
		"fracs_on_np": fracs_on_np,

		# "size_thresholds_list": size_thresholds_list,
		"fractions_tokeep_list": fractions_tokeep_list,
		"steps_to_partially_pretrain": steps_to_partially_pretrain,

		"model_name": model,
		"dataset_choice": dataset,
		"to_mask_epochs": to_mask_epochs,
		"epochs": epochs,
		"N_runs": N_runs,
	}


	save_obj(data_dict, result_file)
	

if __name__ == "__main__":
	args = parser.parse_args()
	main(args)
