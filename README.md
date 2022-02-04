# How many degrees of freedom do we need to train deep networks?

This repository contains source code for the ICLR 2022 paper [***How many degrees of freedom do we need to train deep networks: a loss landscape perspective***](https://openreview.net/forum?id=ChMLTGRjFcU) by Brett W. Larsen, Sanislav Fort, Nic Becker, and Surya Ganguli ([*arXiv version*](https://arxiv.org/abs/2107.05802)). 


This code was developed and tested using `JAX v0.1.74`, `JAXlib v0.1.52`, and `Flax v0.2.0`. The authors intend to update the repository in the future with additional versions of the script that work with the `flax.linen` module.

## Top-Level Scripts
* `burn_in_subspace.py`: Script for random affine subspace and burn-in affine subspace experiments.  To use random affine subspaces, set the parameter `init_iters` to 0.
* `lottery_subspace.py`: Script for lottery subspace experiments
* `lottery_ticket.py`: Script for lottery ticket experiments

## Sub-Functions
* `architectures.py`: Model files
* `data_utils.py`: Functions for saving out data
* `generate_data.py`: Functions to setup datasets for training
* `logging_tools.py`: Setup for logger; generates automatic experiment name with timestamp
* `training_utils.py`: Functions related to projecting to and training in a subspace

## Citation

```
@inproceedings{LaFoBeGa22,
	title={How many degrees of freedom do we need to train deep networks: a loss landscape perspective},
	author={Brett W. Larsen and Stanislav Fort and Nic Becker and Surya Ganguli},
	booktitle={International Conference on Learning Representations},
	year={2022},
	url={https://openreview.net/forum?id=ChMLTGRjFcU}
}
```