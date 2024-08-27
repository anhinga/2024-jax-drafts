import jax
import jax.numpy as jnp
from jax import random
import jax.tree_util as tree_util
import optax

# Example tree-like structure with some fixed weights (e.g., 1) and some to be optimized (None placeholders)
tree_structure = {
    'branch1': jnp.array([1.0, None, None, 1.0]),
    'branch2': {
        'leaf1': jnp.array([None, 1.0, None]),
        'leaf2': jnp.array([1.0, 1.0, None, None])
    }
}

# Initialize random values for optimizable weights
def create_tree(rng_key, tree_structure):
    def initialize_leaf(key, shape):
        return random.normal(key, shape)
    
    def initialize_tree(key, structure):
        leaves, treedef = tree_util.tree_flatten(structure)
        rng_keys = random.split(key, len(leaves))
        
        initialized_leaves = []
        for leaf, rng_key in zip(leaves, rng_keys):
            if leaf is None:
                # Initialize random values where placeholders exist
                initialized_leaves.append(initialize_leaf(rng_key, ()))
            else:
                # Keep fixed values as they are
                initialized_leaves.append(leaf)
        
        return tree_util.tree_unflatten(treedef, initialized_leaves)

    return initialize_tree(rng_key, tree_structure)

# Initialize RNG key
rng_key = random.PRNGKey(0)

# Create the tree with initialized random numbers
random_tree = create_tree(rng_key, tree_structure)

# Define the loss function
def loss_fn(tree):
    # Example loss: sum of squares of all elements in the tree
    return sum([jnp.sum(leaf ** 2) for leaf in tree_util.tree_leaves(tree)])

# Mask to ensure updates only apply to optimizable weights
def mask_tree(tree):
    return tree_util.tree_map(lambda x: x is None, tree_structure)

# Adam optimizer
optimizer = optax.adam(learning_rate=0.001)
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

# Run optimization
for _ in range(10):
    random_tree, opt_state, loss = step(random_tree, opt_state)
    print(f'Loss: {loss}')

