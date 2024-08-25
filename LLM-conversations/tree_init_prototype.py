import jax
import jax.numpy as jnp
from jax import random
import jax.tree_util as tree_util

# Example tree-like structure
def create_tree(rng_key, tree_structure):
    def initialize_leaf(key, shape):
        return random.normal(key, shape)

    rng_keys = random.split(rng_key, len(tree_util.tree_leaves(tree_structure)))
    initialized_leaves = [initialize_leaf(key, shape) for key, shape in zip(rng_keys, tree_util.tree_leaves(tree_structure))]

    return tree_util.tree_unflatten(tree_util.tree_structure(tree_structure), initialized_leaves)

# Define the structure: replace `None` with the shapes of the leaves
tree_structure = {
    'branch1': jnp.zeros((2, 2)),
    'branch2': {
        'leaf1': jnp.zeros((3,)),
        'leaf2': jnp.zeros((4,))
    }
}

# Initialize RNG key
rng_key = random.PRNGKey(0)

# Create the tree with initialized random numbers
random_tree = create_tree(rng_key, tree_structure)
