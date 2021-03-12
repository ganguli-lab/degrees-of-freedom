import numpy as onp #original numpy
import jax.numpy as jnp #jax numpy
import itertools
from jax import jit, grad, random
import jax


def generate_projection(d,D,k_nonzero = None,enforce_no_overlap_if_possible = True):

	M_random = onp.random.normal(loc=0.0,scale=1.0,size=(d,D))

	if k_nonzero is None: #no conditions on axis alignment
		M_now = M_random
	else:
		M_now = onp.zeros((d,D))

		if ((k_nonzero*d <= D) and (enforce_no_overlap_if_possible == True)):
			ids_flat = onp.random.choice(range(D),(k_nonzero*d),replace=False)
			ids_shaped = ids_flat.reshape([d,k_nonzero])
		elif ((k_nonzero*d <= D) and (enforce_no_overlap_if_possible == False)):
			ids_flat = onp.random.choice(range(D),(k_nonzero*d),replace=True)
			ids_shaped = ids_flat.reshape([d,k_nonzero])
		else:
			ids_flat = onp.random.choice(range(D),(k_nonzero*d),replace=True)
			ids_shaped = ids_flat.reshape([d,k_nonzero])

		for i in range(d):
			M_now[i,ids_shaped[i]] = M_random[i,ids_shaped[i]]

	#normalization to unit length of each basis vector
	M_now = M_now / onp.linalg.norm(M_now,axis=-1,keepdims=True)  

	return M_now



## These are the functions required for doing optimization in the hyperplane
@jit
def flatten_leaves(leaves):
	shapes_list = []
	vals_list = []
	for leaf in leaves:
		shapes_list.append(leaf.shape)
		vals_list.append(leaf.reshape([-1]))
	return jnp.concatenate(vals_list),shapes_list

def reform_leaves(vec,shapes_list):
	counter = 0
	leaves = []
	for shape in shapes_list:
		step = jnp.prod(shape)
		leaves.append((vec[counter:counter+step]).reshape(shape))
		counter = counter + step
	return leaves


@jit
def theta_to_flat_params(theta,M,flat_params0):
	return jnp.matmul(theta,M)[0] + flat_params0


def theta_to_paramstree(theta,M,flat_params0,treedef,shapes_list):
	flat_params = theta_to_flat_params(theta,M,flat_params0)
	leaves = reform_leaves(flat_params,shapes_list)
	return jax.tree_unflatten(treedef,leaves)

# Sparse matrix vector multiplication
@jax.partial(jax.jit, static_argnums=(2))
def sp_matmul(A, B, D):
	"""
	Arguments:
		A: (N, M) sparse matrix represented as a tuple (indexes, values)
		B: (M,K) dense matrix
		D: value of N, full weight space dimension
	Returns:
		(N, K) dense matrix
	Modified from: 
		https://gcucurull.github.io/deep-learning/2020/06/03/jax-sparse-matrix-multiplication/
	"""
	assert B.ndim == 2
	rows, cols, values = A
	in_ = B.take(cols.astype('int32'), axis=0)
	prod = in_*values[:, None]
	res = jax.ops.segment_sum(prod, rows.astype('int32'), D)
	return res

#Same functions as above, but using sparse matrix multiplications
@jit
def sparse_theta_to_flat_params(theta,M,flat_params0):	
	MTthetaT = sp_matmul(M,theta.T,flat_params0.shape[0])
	return MTthetaT.T[0] + flat_params0

def sparse_theta_to_paramstree(theta,M,flat_params0,treedef,shapes_list):
	flat_params = sparse_theta_to_flat_params(theta,M,flat_params0)
	leaves = reform_leaves(flat_params,shapes_list)
	return jax.tree_unflatten(treedef,leaves)