import jax
import jax.numpy as jnp
from jax import random
import jax.tree_util as tree_util
import optax

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
            return random.normal(key, ())
        return leaf
    
    leaves, treedef = tree_util.tree_flatten(tree_structure)
    rng_keys = random.split(rng_key, len(leaves))
    
    initialized_leaves = [initialize_leaf(rng_key, leaf) for rng_key, leaf in zip(rng_keys, leaves)]
    
    return tree_util.tree_unflatten(treedef, initialized_leaves)

# Initialize RNG key
rng_key = random.PRNGKey(0)

# Create the tree with initialized random numbers
random_tree = create_tree(rng_key, tree_structure)

# Define the loss function
def loss_fn(tree):
    # Example loss: sum of squares of all elements in the tree
    return sum([leaf ** 2 for leaf in tree_util.tree_leaves(tree)])

# Mask to ensure updates only apply to optimizable weights
def mask_tree(tree):
    return tree_util.tree_map(lambda x: x == SENTINEL, tree_structure)

# Adam optimizer
optimizer = optax.adam(learning_rate=0.1)
opt_state = optimizer.init(random_tree)

# Define the optimization step
@jax.jit
def step(tree, opt_state):
    loss, grads = jax.value_and_grad(loss_fn)(tree)
    
    # Apply mask to gradients
    mask = mask_tree(tree_structure)
    grads = tree_util.tree_map(lambda g, m: g if m else 0, grads, mask)
    
    updates, opt_state = optimizer.update(grads, opt_state)
    new_tree = optax.apply_updates(tree, updates)
    
    return new_tree, opt_state, loss

print(random_tree)

# Run optimization
for _ in range(10):
    random_tree, opt_state, loss = step(random_tree, opt_state)
    print(f'Loss: {loss}')
