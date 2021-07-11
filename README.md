# How many degrees of freedom do we need to train deep networks?

In this repository, we provide code for the following paper:

Brett W. Larsen, Sanislav Fort, Nic Becker, and Surya Ganguli. "How many degrees of freedom do we need to train deep networks: a loss landscape perspective." 2021.

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