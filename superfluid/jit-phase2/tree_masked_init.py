import jax
import jax.numpy as jnp
from jax import random
import jax.tree_util as tree_util

# Define a unique sentinel value that won't appear elsewhere in your data
SENTINEL = -9999.0

# Example tree-like structure with scalar-only leaves
tree_structure = {
    'branch1': {'0': 1.0, '1': SENTINEL, '2': 1.0},
    'branch2': {'leaf1': SENTINEL, 'leaf2': 1.0}
}

# Initialize random values for optimizable weights
def create_tree(rng_key, tree_structure):
    def initialize_leaf(key, leaf):
        if leaf == SENTINEL:
            return 0.1*random.normal(key, ())
        return leaf
    
    leaves, treedef = tree_util.tree_flatten(tree_structure)
    rng_keys = random.split(rng_key, len(leaves))
    
    initialized_leaves = [initialize_leaf(rng_key, leaf) for rng_key, leaf in zip(rng_keys, leaves)]
    
    return tree_util.tree_unflatten(treedef, initialized_leaves)

# Initialize RNG key
rng_key = random.PRNGKey(0)

# Create the tree with initialized random numbers
random_tree = create_tree(rng_key, tree_structure)
